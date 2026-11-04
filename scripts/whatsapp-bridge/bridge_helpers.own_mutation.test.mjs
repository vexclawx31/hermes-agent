import test from 'node:test';
import assert from 'node:assert/strict';

import { ownGroupMutationEvent, parseDeleteRequest } from './bridge_helpers.js';

const GROUP = '120363000000000001@g.us';

function own(message, { fromMe = true, remoteJid = GROUP, id = 'P1', messageTimestamp = 1700000000 } = {}) {
  return { key: { remoteJid, id, fromMe, participant: '' }, message, messageTimestamp };
}

test('parseDeleteRequest accepts group JIDs and bounded ids only', () => {
  assert.deepEqual(parseDeleteRequest({ chatId: GROUP, messageId: 'ABC' }), { chatId: GROUP, messageId: 'ABC' });
  assert.equal(parseDeleteRequest({ chatId: '15550000001@s.whatsapp.net', messageId: 'ABC' }), null);
  assert.equal(parseDeleteRequest({ chatId: 'status@broadcast', messageId: 'ABC' }), null);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: '' }), null);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: 'x'.repeat(129) }), null);
  assert.deepEqual(parseDeleteRequest({ chatId: GROUP, messageId: 'x'.repeat(128) })?.messageId.length, 128);
  assert.equal(parseDeleteRequest({ chatId: GROUP, messageId: 7 }), null);
  assert.equal(parseDeleteRequest(undefined), null);
});

test('own revoke in a group becomes an observer-only event', () => {
  const event = ownGroupMutationEvent(own({ protocolMessage: { type: 0, key: { id: 'T1', remoteJid: GROUP, fromMe: true } } }));
  assert.deepEqual(event, {
    observerOnly: true, ownMutation: 'revoke', chatId: GROUP, isGroup: true, fromMe: true,
    messageId: 'P1', targetMessageId: 'T1', body: '', timestamp: 1700000000,
  });
  // REVOKE is the proto default, so a decoded message may omit the type.
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { key: { id: 'T1' } } })).ownMutation, 'revoke');
});

test('own edit (direct or wrapped by another linked device) carries the new text', () => {
  const direct = ownGroupMutationEvent(own({
    protocolMessage: { type: 14, key: { id: 'T2' }, editedMessage: { conversation: 'fixed text' } },
  }));
  assert.equal(direct.ownMutation, 'edit');
  assert.equal(direct.targetMessageId, 'T2');
  assert.equal(direct.body, 'fixed text');

  const wrapped = ownGroupMutationEvent(own({
    editedMessage: { message: { protocolMessage: { type: 14, key: { id: 'T3' }, editedMessage: { extendedTextMessage: { text: 'v2' } } } } },
  }, { messageTimestamp: { toNumber: () => 1700000001 } }));
  assert.equal(wrapped.ownMutation, 'edit');
  assert.equal(wrapped.body, 'v2');
  assert.equal(wrapped.timestamp, 1700000001);
});

test('anything else from own or foreign messages is not a mutation event', () => {
  const revoke = { protocolMessage: { type: 0, key: { id: 'T1' } } };
  assert.equal(ownGroupMutationEvent(own(revoke, { fromMe: false })), null);
  assert.equal(ownGroupMutationEvent(own(revoke, { remoteJid: '15550000001@s.whatsapp.net' })), null);
  assert.equal(ownGroupMutationEvent(own({ conversation: 'plain own message' })), null);
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 3, key: { id: 'T1' } } })), null); // ephemeral setting
  assert.equal(ownGroupMutationEvent(own({ protocolMessage: { type: 0 } })), null); // no target id
  assert.equal(ownGroupMutationEvent(own(revoke, { id: '' })), null);
  assert.equal(ownGroupMutationEvent({}), null);
});
