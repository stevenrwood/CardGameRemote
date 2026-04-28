"""Build / redact the JSON document polled by ``/table/state``.

The /table view is what Rodney watches over Teams: his own hand
in full, every other player as a down-count + Brio up cards, the
challenge / draw / verify state, the tail of the table log. This
module owns the shape of that document and the helpers used to
build it (card parsing, best-hand evaluation).

Cross-module helpers (challenge / game-meta) are imported
lazily inside ``_build_table_state`` so this module stays
import-cheap and load-order-flexible.
"""

import re


_CARD_NAME_RE = re.compile(
    r"^\s*(Ace|King|Queen|Jack|10|[2-9])\s+of\s+(Hearts|Diamonds|Clubs|Spades)\s*$",
    re.IGNORECASE,
)
_RANK_CANON = {"ACE": "A", "KING": "K", "QUEEN": "Q", "JACK": "J"}
_SUIT_LETTER = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}
_SUIT_LETTER_CODE = {"clubs": "c", "diamonds": "d", "hearts": "h", "spades": "s"}


def _parse_card_any(text):
    """Parse either 'King of Hearts' or 'Kh' / '10s' into {rank, suit} or None."""
    if not text:
        return None
    text = str(text).strip()
    m = _CARD_NAME_RE.match(text)
    if m:
        rank = m.group(1).upper()
        rank = _RANK_CANON.get(rank, rank)
        return {"rank": rank, "suit": m.group(2).lower()}
    m = re.match(r"^(10|[2-9JQKA])([hdcs])$", text, re.IGNORECASE)
    if m:
        return {"rank": m.group(1).upper(), "suit": _SUIT_LETTER[m.group(2).lower()]}
    return None


def _best_hand_for_cards(cards, ge):
    """Given a list of card dicts ({rank, suit, ...}), compute the best
    poker hand using the current game's wild ranks and return
    {"label": ..., "codes": [...]} for the /table UI. codes are short
    card codes ("Ah", "10s") in best-hand order so the client can reorder
    its card row. Returns None if fewer than 2 cards or evaluation fails.
    """
    tuples = []
    code_by_id = {}
    for i, c in enumerate(cards):
        rank = c.get("rank")
        suit = c.get("suit")
        if not rank or not suit:
            continue
        tuples.append((rank, suit))
        code_by_id[i] = f"{rank}{_SUIT_LETTER_CODE.get(suit, (suit or '?')[0])}"
    if len(tuples) < 2:
        return None
    try:
        from poker_hands import best_hand
    except Exception:
        return None
    try:
        wilds = list(getattr(ge, "wild_ranks", []) or [])
        result = best_hand(tuples, wild_ranks=wilds)
    except Exception:
        return None
    codes = []
    for bc in result.cards:
        suit_letter = _SUIT_LETTER_CODE.get(bc.suit, (bc.suit or "?")[0])
        codes.append(f"{bc.rank}{suit_letter}")
    return {"label": result.label, "codes": codes, "category": result.category}


def _build_table_state(s):
    """Produce the JSON doc that /table/state returns.

    Rodney sees his hand in full. Every other player is just a down-count
    plus Brio up-card scans. The log is the tail of table_log.
    """
    # Lazy imports — challenge.py + game_meta.py would cause a
    # circular import if pulled at module load. By the time
    # _build_table_state actually runs, both are fully loaded.
    from games.challenge import (
        _challenge_can_mark,
        _challenge_required_cards,
        _game_is_challenge,
    )
    from game_meta import (
        _game_has_draw_phase,
        _max_draw_for_game,
        _total_card_rounds,
        _total_draw_phases,
    )

    ge = s.game_engine
    current_game = ge.current_game.name if ge.current_game else None

    # Accumulate up-card history by player from console_hand_cards (populated
    # when the dealer confirms each up-card round). Fall back to the latest
    # zone scan for any player missing from history — useful between rounds
    # before the dealer has hit Confirm.
    up_by_player = {}
    for entry in s.console_hand_cards:
        name = entry.get("player")
        parsed = _parse_card_any(entry.get("card", ""))
        if name and parsed:
            up_by_player.setdefault(name, []).append(
                {"rank": parsed["rank"], "suit": parsed["suit"], "round": entry.get("round")}
            )

    players = []
    for p in ge.players:
        if p.name not in s.console_active_players:
            continue
        up_cards = list(up_by_player.get(p.name, []))
        if not up_cards and s.monitor:
            # Only show latest scan if we don't already have history for this player.
            latest_txt = s.monitor.last_card.get(p.name, "")
            latest_parsed = _parse_card_any(latest_txt)
            if latest_parsed:
                details = s.monitor.recognition_details.get(p.name, {})
                conf = details.get("yolo_conf")
                cur = {"rank": latest_parsed["rank"], "suit": latest_parsed["suit"]}
                if conf is not None:
                    cur["confidence"] = round(float(conf), 2)
                up_cards.append(cur)

        freezes_n = s.freezes.get(p.name, 0)
        entry = {
            "name": p.name,
            "position": p.position,
            "is_dealer": p.is_dealer,
            "is_remote": p.is_remote,
            "folded": p.name in s.folded_players,
            "freezes": freezes_n,
            "frozen": freezes_n >= 3,
        }
        if p.is_remote:
            # Rodney's hand = only down-card slots that have been recognized
            # and validated (rodney_downs). Tentative slot_pending guesses
            # are shown in the verify modal instead, not as cards in hand.
            # If Rodney flipped one of his downs face-up (7/27 2-down), the
            # card remains in rodney_downs (so Pi counting still works) but
            # we render it here as an up-card instead of a down.
            flipped_slot = (s.rodney_flipped_up or {}).get("slot")
            hand = []
            for slot_num in sorted(s.rodney_downs.keys()):
                if slot_num == flipped_slot:
                    continue
                d = s.rodney_downs[slot_num]
                hand.append({"type": "down", "rank": d["rank"],
                             "suit": d["suit"], "slot": slot_num,
                             "confidence": d.get("confidence")})
            # Challenge round 3 displaces the round-2 cards from
            # slots 4 and 5 (they sit face-down in front of Rodney
            # off-scanner). Render them too so /table shows all 7.
            # No slot field on overflow entries — they aren't
            # markable and shouldn't collide with scanner slots.
            # NOTE: use a distinct loop variable; reusing `entry`
            # would clobber the outer player-dict reference (the
            # one we still write hand/best_hand into below).
            for ov_entry in (s.rodney_overflow or []):
                card = ov_entry.get("card") or {}
                if card.get("rank") and card.get("suit"):
                    hand.append({
                        "type": "down",
                        "rank": card["rank"],
                        "suit": card["suit"],
                        "confidence": card.get("confidence"),
                        "off_scanner": True,
                    })
            if s.rodney_flipped_up:
                fu = s.rodney_flipped_up
                # Don't duplicate once Brio picks it up via a zone scan.
                already = any(
                    c.get("rank") == fu["rank"] and c.get("suit") == fu["suit"]
                    for c in up_cards
                )
                if not already:
                    hand.append({"type": "up", "rank": fu["rank"], "suit": fu["suit"]})
            for c in up_cards:
                hand.append({"type": "up", **c})
            entry["hand"] = hand
            entry["best_hand"] = _best_hand_for_cards(hand, ge)
        else:
            # Dealer deals the same card-type to every player in each round,
            # so every non-folded player holds as many downs as Rodney has
            # validated. In 7/27 (2-down) once Rodney has flipped, every
            # local player has also flipped one of their two — subtract the
            # flipped card from the visible down-count.
            down_count = len(s.rodney_downs)
            if s.rodney_flipped_up:
                down_count = max(0, down_count - 1)
            entry["down_count"] = down_count
            entry["up_cards"] = up_cards
            entry["best_hand"] = _best_hand_for_cards(up_cards, ge)
        players.append(entry)

    # Console flow doesn't advance game_engine.phase_index, so derive the
    # round counter from console_up_round (confirmed up rounds) + down cards
    # Rodney has actually received — including pending scans so a yet-to-be-
    # verified card still advances the counter.
    active_down_slots = set(s.rodney_downs.keys()) | set(s.slot_pending.keys())
    current_round = s.console_up_round + len(active_down_slots)
    total_rounds = _total_card_rounds(ge)
    # Open-ended games (e.g. 7/27) report total=0 so the UI drops "of N".
    if ge.current_game is not None:
        has_hit_round = any(
            ph.type.value == "hit_round" and ph.card_type == "up"
            for ph in ge.current_game.phases
        )
        if has_hit_round:
            total_rounds = 0

    doc = {
        "version": s.table_state_version,
        "viewer": next((p.name for p in ge.players if p.is_remote), "Rodney"),
        "game": {
            "name": current_game or "",
            "round": getattr(ge, "draw_round", 0),
            "wild_label": ge.wild_label or "",
            "wild_ranks": list(getattr(ge, "wild_ranks", []) or []),
            "current_round": current_round,
            "total_rounds": total_rounds,
            "state": getattr(ge.state, "value", str(ge.state)),
        },
        "dealer": ge.get_dealer().name,
        "current_player": None,
        "players": players,
        "log": list(s.table_log[-30:]),
        "pending_verify": s.pending_verify,
        "flip_choice": None,
        "guided_deal": (
            dict(s.guided_deal) if s.guided_deal is not None else None
        ),
        "draw": {
            # Multi-draw games (3 Toed Pete): rodney_draws_done counts how
            # many draws are behind us; we can mark and request again as
            # long as more DRAW phases remain and the current draw has not
            # been taken yet.
            "can_mark": (
                (
                    _game_has_draw_phase(ge)
                    and s.rodney_draws_done < _total_draw_phases(ge)
                    and not s.rodney_drew_this_hand
                    and s.console_state in ("dealing", "betting", "draw")
                )
                or _challenge_can_mark(s, ge)
            ),
            "can_request": (
                s.console_state == "draw"
                and s.rodney_draws_done < _total_draw_phases(ge)
                and not s.rodney_drew_this_hand
            ),
            "max_marks": (
                _challenge_required_cards(s)
                if _challenge_can_mark(s, ge)
                else _max_draw_for_game(ge, s.rodney_draws_done, s)
            ),
            "marked_slots": sorted(s.rodney_marked_slots),
            "drew_this_hand": s.rodney_drew_this_hand,
            "draws_done": s.rodney_draws_done,
            "total_draws": _total_draw_phases(ge),
        },
    }
    # Challenge-game block — populated only for Challenge variants.
    if _game_is_challenge(ge) and s.challenge_round_index is not None:
        round_label = None
        try:
            gname = ge.current_game.name if ge.current_game else ""
            labels = {
                "High, Low, High": ["High", "Low", "High"],
                "Low, High, Low": ["Low", "High", "Low"],
                "Low, Low, High": ["Low", "Low", "High"],
            }.get(gname)
            if labels and s.challenge_round_index < len(labels):
                round_label = labels[s.challenge_round_index]
        except Exception:
            pass
        doc["challenge"] = {
            "round_index": s.challenge_round_index,
            "round_label": round_label,
            "required_cards": _challenge_required_cards(s),
            "shuffle_count": s.challenge_shuffle_count,
            "pot_cents": s.pot_cents,
            "per_player": {
                nm: {
                    "went_out": st["went_out"],
                    "passes": int(st.get("passes", 0)),
                    "out_round": st["out_round"],
                    "out_slots": list(st["out_slots"]),
                } for nm, st in s.challenge_per_player.items()
            },
            "rodney_out_slots": list(s.rodney_out_slots),
            "rodney_overflow": list(s.rodney_overflow),
        }
    # Per-game decorations: the game class adds its own fields to the
    # document (7/27 injects values_7_27 per player and the flip-choice
    # prompt; base class / stud / draw are no-ops).
    impl = s.current_game_impl
    if impl is not None:
        impl.decorate_table_players(players, s)
        impl.decorate_table_state(doc, s)
    return doc


def _table_state_bump(s):
    """Call when something observable changes so polling clients re-render."""
    s.table_state_version += 1


def _redact_remote_downs(doc):
    """Hide the remote player's hole cards in a /table/state document
    so a viewer who isn't the remote player can't read his hand.

    Mirrors what local players actually see at the table: down cards
    show as face-down placeholders (preserving the count), and any
    derived totals that incorporated those cards (poker best_hand,
    7/27 values_7_27) are dropped.
    """
    for entry in doc.get("players", []) or []:
        if not entry.get("is_remote"):
            continue
        new_hand = []
        for card in entry.get("hand", []) or []:
            if card.get("type") == "down":
                new_hand.append({"type": "down", "hidden": True})
            else:
                new_hand.append(card)
        if "hand" in entry:
            entry["hand"] = new_hand
        entry.pop("best_hand", None)
        entry.pop("values_7_27", None)
