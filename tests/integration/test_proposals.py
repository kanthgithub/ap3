"""Live agents on ports 19140-19156.

Remember keeps the context it was given. inputs['send'] is the only path that puts an envelope on the wire.
"""

import asyncio
import time
from datetime import datetime, timezone

import pytest

from ap3 import Operation, OperationResult
from ap3.a2a.agent import PrivacyAgent
from ap3.a2a.client import PeerClient
from ap3.services.commitment import CommitmentMetadataSystem
from ap3.signing.primitives import generate_keypair
from ap3.types import (
    CommitmentMetadata,
    DataFormat,
    DataFreshness,
    DataStructure,
    Industry,
    PrivacyProtocolError,
)


def commitment(agent_id, structure, **extra):
    values = dict(
        agent_id=agent_id,
        commitment_id=f"{agent_id}_data",
        data_structure=structure,
        data_format=DataFormat.STRUCTURED,
        entry_count=6,
        field_count=1,
        estimated_size_mb=0.001,
        last_updated=datetime.now(timezone.utc).isoformat(),
        data_freshness=DataFreshness.REAL_TIME,
        industry=Industry.FINANCE,
    )
    values.update(extra)
    return CommitmentMetadata(**values)


def tampered(commitment, private_key):
    signed = commitment.model_copy(
        update={"signature": CommitmentMetadataSystem.sign_commitment(commitment, private_key)}
    )
    return signed.model_copy(update={"entry_count": signed.entry_count + 1})


class Remember(Operation):
    operation_id = "proposal.remember"
    operation_type = "PSI"

    def __init__(self):
        super().__init__()
        self.contexts = []
        self.received = False

    def on_start(self, role, inputs, config, context):
        self.contexts.append(context)
        if inputs.get("send"):
            return OperationResult(next_state={}, outgoing={"phase": "ping", "message": "hi"})
        return OperationResult(done=True, result={"ok": True})

    def on_process(self, role, state, message, config, context):
        if role == "receiver":
            self.received = True
            return OperationResult(done=True, outgoing={"phase": "pong", "message": "hi"})
        return OperationResult(done=True, result={"ok": True})


def agent(name, port, role, operation, private_key, public_key, structure, bad=False, peer_client=None, **extra):
    meta = commitment(name, structure, **extra)
    if bad:
        meta = tampered(meta, private_key)
    return PrivacyAgent(
        name=name,
        description=name,
        card_url=f"http://localhost:{port}",
        host="localhost",
        port=port,
        role=role,
        operation=operation,
        commitment=meta,
        private_key=private_key,
        public_key=public_key,
        peer_client=peer_client,
        allow_private_initiator_urls=True,
    )


@pytest.mark.asyncio
async def test_honest_card_verifies_and_is_handed_to_the_operation():
    """Given two honest cards. When A runs one intent. Then the operation sees C's signed commitment."""
    a_key = generate_keypair()
    c_key = generate_keypair()
    remember = Remember()
    asking = agent("agent_a", 19140, "ap3_initiator", remember, *a_key, DataStructure.CUSTOMER_LIST)
    answering = agent("agent_c", 19141, "ap3_receiver", Remember(), *c_key, DataStructure.BLACKLIST)
    async with asking.serving(), answering.serving():
        result = await asking.run_intent(peer_url="http://localhost:19141", inputs={})
    assert result.result_data is not None
    seen = remember.contexts[0]["peer_commitments"][0]
    assert seen.agent_id == "agent_c"
    assert seen.signature


@pytest.mark.asyncio
async def test_changed_peer_commitment_is_refused_before_sending():
    """Given C's card signed, then entry_count bumped. When A calls run_intent. Then COMMITMENT_SIGNATURE_INVALID, and C is not entered."""
    a_key = generate_keypair()
    c_key = generate_keypair()
    receiver_op = Remember()
    asking = agent("agent_a", 19142, "ap3_initiator", Remember(), *a_key, DataStructure.CUSTOMER_LIST)
    answering = agent(
        "agent_c", 19143, "ap3_receiver", receiver_op, *c_key, DataStructure.BLACKLIST, bad=True
    )
    async with asking.serving(), answering.serving():
        with pytest.raises(PrivacyProtocolError) as caught:
            await asking.run_intent(peer_url="http://localhost:19143", inputs={})
    assert caught.value.error.error_code == "COMMITMENT_SIGNATURE_INVALID"
    assert receiver_op.received is False


@pytest.mark.asyncio
async def test_changed_initiator_commitment_is_refused_by_the_receiver():
    """Given A's card signed, then entry_count bumped. When A sends one envelope. Then C refuses, and C's operation is not entered."""
    a_key = generate_keypair()
    c_key = generate_keypair()
    receiver_op = Remember()
    asking = agent(
        "agent_a", 19144, "ap3_initiator", Remember(), *a_key, DataStructure.CUSTOMER_LIST, bad=True
    )
    answering = agent("agent_c", 19145, "ap3_receiver", receiver_op, *c_key, DataStructure.BLACKLIST)
    async with asking.serving(), answering.serving():
        with pytest.raises(PrivacyProtocolError) as caught:
            await asking.run_intent(peer_url="http://localhost:19145", inputs={"send": True})
    assert caught.value.error.error_code == "COMMITMENT_SIGNATURE_INVALID"
    assert receiver_op.received is False


@pytest.mark.asyncio
async def test_run_intents_reports_a_stopped_peer_on_its_own():
    """Given B and C up, D down. When A calls run_intents for all three. Then B and C return a result and D is an exception."""
    a_key = generate_keypair()
    b_key = generate_keypair()
    c_key = generate_keypair()
    asking = agent("agent_a", 19146, "ap3_initiator", Remember(), *a_key, DataStructure.CUSTOMER_LIST)
    bee = agent("agent_b", 19147, "ap3_receiver", Remember(), *b_key, DataStructure.BLACKLIST)
    cee = agent("agent_c", 19148, "ap3_receiver", Remember(), *c_key, DataStructure.BLACKLIST)
    urls = ["http://localhost:19147", "http://localhost:19148", "http://localhost:19149"]
    async with asking.serving(), bee.serving(), cee.serving():
        results = await asking.run_intents(peer_urls=urls, inputs={})
    assert set(results) == set(urls)
    assert not isinstance(results[urls[0]], BaseException)
    assert not isinstance(results[urls[1]], BaseException)
    assert isinstance(results[urls[2]], BaseException)


@pytest.mark.asyncio
async def test_run_intents_times_out_a_slow_peer(monkeypatch):
    """Given B answers and C never replies. When A calls run_intents with timeout_s. Then C is PEER_TIMEOUT, B is a result, and A keeps no open session."""
    a_key = generate_keypair()
    b_key = generate_keypair()
    c_key = generate_keypair()
    remember = Remember()
    asking = agent("agent_a", 19152, "ap3_initiator", remember, *a_key, DataStructure.CUSTOMER_LIST)
    bee = agent("agent_b", 19153, "ap3_receiver", Remember(), *b_key, DataStructure.BLACKLIST)
    cee = agent("agent_c", 19154, "ap3_receiver", Remember(), *c_key, DataStructure.BLACKLIST)
    release = asyncio.Event()

    async def stall(envelope):
        await release.wait()

    monkeypatch.setattr(cee, "handle_envelope", stall)
    urls = ["http://localhost:19153", "http://localhost:19154"]
    async with asking.serving(), bee.serving(), cee.serving():
        started = time.monotonic()
        results = await asking.run_intents(peer_urls=urls, inputs={"send": True}, timeout_s=0.5)
        took = time.monotonic() - started
        release.set()
    assert not isinstance(results[urls[0]], BaseException)
    assert results[urls[1]].error.error_code == "PEER_TIMEOUT"
    assert took < 2
    assert not remember._sessions


@pytest.mark.asyncio
async def test_run_intents_caps_open_sessions(monkeypatch):
    """Given five URLs and max_concurrency=2. When run_intents runs. Then at most two run at once, and all five return."""
    asking = agent("agent_a", 19155, "ap3_initiator", Remember(), *generate_keypair(), DataStructure.CUSTOMER_LIST)
    open_now = 0
    most = 0

    async def fake(*, peer_url, **_):
        nonlocal open_now, most
        open_now += 1
        most = max(most, open_now)
        await asyncio.sleep(0.05)
        open_now -= 1
        return peer_url

    monkeypatch.setattr(asking._core, "run_intent", fake)
    urls = [f"http://peer-{i}.example" for i in range(5)]
    results = await asking.run_intents(peer_urls=urls, inputs={}, max_concurrency=2)
    assert most == 2
    assert results == {url: url for url in urls}


@pytest.mark.asyncio
async def test_run_intents_refuses_duplicate_urls():
    """Given the same URL twice. When run_intents is called. Then ValueError, because results are keyed by URL."""
    asking = agent("agent_a", 19156, "ap3_initiator", Remember(), *generate_keypair(), DataStructure.CUSTOMER_LIST)
    with pytest.raises(ValueError):
        await asking.run_intents(peer_urls=["http://peer.example", "http://peer.example"], inputs={})


@pytest.mark.asyncio
async def test_update_commitment_is_what_a_refresh_reads():
    """Given C serving v=1. When C publishes v=2. Then a cached read still shows v=1, and refresh_peer=True shows v=2."""
    a_key = generate_keypair()
    c_key = generate_keypair()
    remember = Remember()
    asking = agent(
        "agent_a",
        19150,
        "ap3_initiator",
        remember,
        *a_key,
        DataStructure.CUSTOMER_LIST,
        peer_client=PeerClient(force_refresh_interval_s=0),
    )
    answering = agent(
        "agent_c",
        19151,
        "ap3_receiver",
        Remember(),
        *c_key,
        DataStructure.BLACKLIST,
        operation_params={"v": "1"},
    )
    async with asking.serving(), answering.serving():
        await asking.run_intent(peer_url="http://localhost:19151", inputs={})
        answering.update_commitment(
            commitment("agent_c", DataStructure.BLACKLIST, operation_params={"v": "2"})
        )
        await asking.run_intent(peer_url="http://localhost:19151", inputs={})
        await asking.run_intent(peer_url="http://localhost:19151", inputs={}, refresh_peer=True)
    versions = [ctx["peer_commitments"][0].operation_params["v"] for ctx in remember.contexts]
    assert versions == ["1", "1", "2"]
