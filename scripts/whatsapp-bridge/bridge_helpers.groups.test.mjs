import test from 'node:test';
import assert from 'node:assert/strict';

import { createDeliveryStatusStore, summarizeGroups } from './bridge_helpers.js';

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
  assert.equal(store.update('unknown', 3), false);
  store.track('m1');
  assert.equal(store.get('m1').status, 'pending');
  store.update('m1', 3);
  store.update('m1', 2);
  assert.equal(store.get('m1').status, 'delivery_ack');
  store.track('m2');
  store.track('m3');
  assert.equal(store.get('m1'), null);
  assert.equal(store.get('m3').status, 'pending');
});
