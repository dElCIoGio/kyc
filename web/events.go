package main

import (
	"sync"
	"time"
)

type eventHub struct {
	mu   sync.Mutex
	subs map[string]map[chan struct{}]struct{}
}

func newEventHub() *eventHub { return &eventHub{subs: make(map[string]map[chan struct{}]struct{})} }

func (hub *eventHub) subscribe(token string) (<-chan struct{}, func()) {
	channel := make(chan struct{}, 1)
	hub.mu.Lock()
	if hub.subs[token] == nil { hub.subs[token] = make(map[chan struct{}]struct{}) }
	hub.subs[token][channel] = struct{}{}
	hub.mu.Unlock()
	return channel, func() {
		hub.mu.Lock()
		defer hub.mu.Unlock()
		if subscribers := hub.subs[token]; subscribers != nil {
			delete(subscribers, channel)
			if len(subscribers) == 0 { delete(hub.subs, token) }
		}
	}
}

func (hub *eventHub) publish(token string) {
	hub.mu.Lock()
	defer hub.mu.Unlock()
	for channel := range hub.subs[token] {
		select { case channel <- struct{}{}: default: }
	}
}

type eventDeduper struct { mu sync.Mutex; seen map[string]time.Time }
func newEventDeduper() *eventDeduper { return &eventDeduper{seen: make(map[string]time.Time)} }
func (deduper *eventDeduper) first(id string) bool {
	deduper.mu.Lock(); defer deduper.mu.Unlock()
	now := time.Now()
	for key, expiry := range deduper.seen { if !expiry.After(now) { delete(deduper.seen, key) } }
	if _, ok := deduper.seen[id]; ok { return false }
	deduper.seen[id] = now.Add(24 * time.Hour)
	return true
}
