import test from 'node:test';
import assert from 'node:assert/strict';

import { GROUP_JID_RE, createDeliveryStatusStore, createHistoryQueue, isArchivableHistoryMessage, receiptStatus, summarizeGroups } from './bridge_helpers.js';

test('summarizeGroups keeps ids, subject and participant ids only, sorted', () => {
  const groups = summarizeGroups({
    'b@g.us': { id: 'b@g.us', subject: 'B', participants: [{ id: '1@lid', phoneNumber: '1@s.whatsapp.net', admin: 'admin', imgUrl: 'x', notify: 'n' }] },
    'a@g.us': { id: 'a@g.us', subject: 'A', announce: true, participants: [{ id: '2@s.whatsapp.net', admin: null }, null] },
    'x@s.whatsapp.net': { id: 'x@s.whatsapp.net', subject: 'not a group' },
  });
  assert.deepEqual(groups.map((g) => g.id), ['a@g.us', 'b@g.us']);
  assert.deepEqual(groups[1].participants, [{ id: '1@lid', phoneNumber: '1@s.whatsapp.net', lid: null, admin: 'admin' }]);
  assert.equal(groups[0].announce, true);
  assert.deepEqual(groups[0].participants, [{ id: '2@s.whatsapp.net', phoneNumber: null, lid: null, admin: null }]);
});

test('delivery status store tracks only sent ids and never regresses', () => {
  const store = createDeliveryStatusStore(2);
  assert.equal(store.update('', 3), false);
  store.update('early', 2); // ack before /send returned
  assert.equal(store.get('early').status, 'server_ack');
  store.track('early');
  assert.equal(store.get('early').status, 'server_ack');
  store.update('early', 4);
  store.update('early', 0);
  assert.equal(store.get('early').status, 'read');
  store.track('m1');
  assert.equal(store.get('m1').status, 'pending');
  store.update('m1', 3);
  store.update('m1', 2);
  assert.equal(store.get('m1').status, 'delivery_ack');
  store.track('m2');
  store.track('m3');
  assert.equal(store.get('m1'), null);
  assert.equal(store.get('early'), null);
  assert.equal(store.get('m3').status, 'pending');
});

test('group receipts map to delivered/read and ignore empty receipts', () => {
  assert.equal(receiptStatus({ receiptTimestamp: 5 }), 3);
  assert.equal(receiptStatus({ receiptTimestamp: 5, readTimestamp: 6 }), 4);
  assert.equal(receiptStatus({ playedTimestamp: 7 }), 4);
  assert.equal(receiptStatus({}), null);
  assert.equal(receiptStatus(null), null);
  // Our own phone's sender receipt is not a recipient read.
  assert.equal(receiptStatus({ userJid: '111:3@s.whatsapp.net', readTimestamp: 1 }, ['111@s.whatsapp.net']), null);
  assert.equal(receiptStatus({ userJid: '222@s.whatsapp.net', readTimestamp: 1 }, ['111@s.whatsapp.net']), 4);
});

test('delivery store exposes has() so receipts never create entries', () => {
  const store = createDeliveryStatusStore(4);
  assert.equal(store.has('x'), false);
  store.track('x');
  assert.equal(store.has('x'), true);
});

test('history filters and group JID validation', () => {
  assert.equal(isArchivableHistoryMessage({ message: { conversation: 'hi' } }), true);
  assert.equal(isArchivableHistoryMessage({ message: { pollUpdateMessage: {} } }), false);
  assert.equal(isArchivableHistoryMessage({ message: { protocolMessage: {} } }), false);
  assert.equal(isArchivableHistoryMessage({}), false);
  assert.equal(GROUP_JID_RE.test('120363000000000001@g.us'), true);
  assert.equal(GROUP_JID_RE.test('x@g.us'), false);
});

test('history queue is bounded and drained in slices', () => {
  const q = createHistoryQueue(3);
  for (const i of [1, 2, 3, 4, 5]) q.push(i);
  assert.equal(q.size, 3);
  assert.deepEqual(q.take(2), [3, 4]);
  assert.deepEqual(q.take(10), [5]);
  assert.deepEqual(q.take(1), []);
  assert.equal(q.dropped, 2);
});
