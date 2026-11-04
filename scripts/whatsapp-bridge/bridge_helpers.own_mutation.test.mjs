import test from 'node:test';
import assert from 'node:assert/strict';

import {
  BRIDGE_CAPABILITIES,
  createDeleteHandler,
  ownGroupMutationEvent,
  parseDeleteRequest,
  queueOwnGroupMutation,
} from './bridge_helpers.js';

const GROUP = '120363000000000001@g.us';
const OWN_TARGET = { id: 'T1', remoteJid: GROUP, fromMe: true };

function own(message, { fromMe = true, remoteJid = GROUP, id = 'P1', messageTimestamp = 1700000000 } = {}) {
  return { key: { remoteJid, id, fromMe, participant: '' }, message, messageTimestamp };
}

function fakeRes() {
  return {
    statusCode: 200,
    body: undefined,
    status(code) { this.statusCode = code; return this; },
    json(body) { this.body = body; return this; },
  };
}

test('parseDeleteRequest accepts group JIDs and bounded ids only', () => {
  assert.deepEqual(parseDeleteRequest({ chatId: GROUP, messageId: 'ABC' }), { chatId: GROUP, messageId: 'ABC' });
  assert.equal(parseDeleteRequest({ chatId: '15550000001@s.whatsapp.net', messageId: 'ABC' }), null);
  assert.equal(parseDeleteRequest({ chatId: 'status@broadcast', messageId: 'ABC' }), null);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: '' }), null);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: 'x'.repeat(129) }), null);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: 'x'.repeat(128) })?.messageId.length, 128);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: 7 }), null);
  assert.equal(parseDeleteRequest(undefined), null);
});

test('own revoke of an own group message becomes an observer-only event without messageId', () => {
  const event = ownGroupMutationEvent(own({ protocolMessage: { type: 0, key: OWN_TARGET } }));
  assert.deepEqual(event, {
    observerOnly: true, ownMutation: 'revoke', chatId: GROUP, isGroup: true, fromMe: true,
    targetMessageId: 'T1', body: '', timestamp: 1700000000,
  });
  assert.equal('messageId' in event, false);
  // REVOKE is the proto default, so a decoded message may omit the type.
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { key: OWN_TARGET } })).ownMutation, 'revoke');
});

test('admin revoke or edit of another member message emits nothing', () => {
  const other = { id: 'T7', remoteJid: GROUP, fromMe: false, participant: '15550000009@s.whatsapp.net' };
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 0, key: other } })), null);
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 0, key: { id: 'T7' } } })), null); // fromMe missing
  assert.equal(ownGroupMutationEvent(own({
    protocolMessage: { type: 14, key: other, editedMessage: { conversation: 'x' } },
  })), null);
});

test('own edit (direct or wrapped by another linked device) carries the new text', () => {
  const direct = ownGroupMutationEvent(own({
    protocolMessage: { type: 14, key: { ...OWN_TARGET, id: 'T2' }, editedMessage: { conversation: 'fixed text' } },
  }));
  assert.equal(direct.ownMutation, 'edit');
  assert.equal(direct.targetMessageId, 'T2');
  assert.equal(direct.body, 'fixed text');

  const wrapped = ownGroupMutationEvent(own({
    editedMessage: { message: { protocolMessage: { type: 14, key: { ...OWN_TARGET, id: 'T3' }, editedMessage: { extendedTextMessage: { text: 'v2' } } } } },
  }, { messageTimestamp: { toNumber: () => 1700000001 } }));
  assert.equal(wrapped.ownMutation, 'edit');
  assert.equal(wrapped.body, 'v2');
  assert.equal(wrapped.timestamp, 1700000001);
});

test('own revoke and edit in a disappearing-message group are unwrapped', () => {
  const revoke = ownGroupMutationEvent(own({ ephemeralMessage: { message: { protocolMessage: { type: 0, key: OWN_TARGET } } } }));
  assert.equal(revoke.ownMutation, 'revoke');
  const edit = ownGroupMutationEvent(own({
    ephemeralMessage: { message: { editedMessage: { message: { protocolMessage: {
      type: 14, key: OWN_TARGET, editedMessage: { conversation: 'eph' } } } } } },
  }));
  assert.equal(edit.ownMutation, 'edit');
  assert.equal(edit.body, 'eph');
  const viewOnce = ownGroupMutationEvent(own({ viewOnceMessage: { message: { protocolMessage: { type: 0, key: OWN_TARGET } } } }));
  assert.equal(viewOnce.ownMutation, 'revoke');
});

test('anything else from own or foreign messages is not a mutation event', () => {
  const revoke = { protocolMessage: { type: 0, key: OWN_TARGET } };
  assert.equal(ownGroupMutationEvent(own(revoke, { fromMe: false })), null);
  assert.equal(ownGroupMutationEvent(own(revoke, { remoteJid: '15550000001@s.whatsapp.net' })), null);
  assert.equal(ownGroupMutationEvent(own({ conversation: 'plain own message' })), null);
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 3, key: OWN_TARGET } })), null); // ephemeral setting
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 0 } })), null); // no target key
  assert.equal(ownGroupMutationEvent({}), null);
});

test('queueOwnGroupMutation queues only mutations and stays bounded', () => {
  const queue = [{ old: 1 }, { old: 2 }];
  assert.equal(queueOwnGroupMutation(own({ conversation: 'plain' }), queue, 2), null);
  assert.equal(queue.length, 2);
  const event = queueOwnGroupMutation(own({ protocolMessage: { type: 0, key: OWN_TARGET } }), queue, 2);
  assert.equal(event.ownMutation, 'revoke');
  assert.deepEqual(queue, [{ old: 2 }, event]);
});

test('health capabilities stay an object and advertise delete and own-mutation events', () => {
  assert.equal(Array.isArray(BRIDGE_CAPABILITIES), false);
  for (const key of ['outboundMentions', 'groupDirectory', 'deliveryStatus', 'historyRequest',
    'send', 'edit', 'sendMedia', 'sendPoll', 'delete', 'ownMutationEvents']) {
    assert.equal(BRIDGE_CAPABILITIES[key], true, key);
  }
});

test('/delete handler: 503, 400, revoke payload, 500 refusal, 504 unknown', async () => {
  const sent = [];
  let connected = false;
  let failure = null;
  const handler = createDeleteHandler({
    isConnected: () => connected,
    send: async (chatId, payload) => {
      sent.push([chatId, payload]);
      if (failure) throw failure;
      return { key: { id: 'R1' } };
    },
  });
  const call = async (body) => { const res = fakeRes(); await handler({ body }, res); return res; };

  let res = await call({ chatId: GROUP, messageId: 'M1' });
  assert.equal(res.statusCode, 503);
  connected = true;
  res = await call({ chatId: '15550000001@s.whatsapp.net', messageId: 'M1' });
  assert.equal(res.statusCode, 400);
  assert.equal(sent.length, 0);

  res = await call({ chatId: GROUP, messageId: 'M1' });
  assert.deepEqual([res.statusCode, res.body], [200, { success: true, messageId: 'M1' }]);
  assert.deepEqual(sent[0], [GROUP, { delete: { remoteJid: GROUP, id: 'M1', fromMe: true } }]);

  failure = new Error('not allowed');
  res = await call({ chatId: GROUP, messageId: 'M1' });
  assert.deepEqual([res.statusCode, res.body], [500, { error: 'not allowed' }]);

  failure = Object.assign(new Error('sendMessage timed out after 30s'), { code: 'SEND_TIMEOUT' });
  res = await call({ chatId: GROUP, messageId: 'M1' });
  assert.deepEqual([res.statusCode, res.body], [504, { error: 'sendMessage timed out after 30s', outcome: 'unknown' }]);
});
