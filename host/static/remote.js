// Remote player WebSocket client

const SUIT_SYMBOLS = {
    hearts: "\u2665", diamonds: "\u2666",
    clubs: "\u2663", spades: "\u2660"
};

let ws = null;
let hand = [];          // Array of slot objects
let selectedSlots = []; // Currently selected slot numbers
let actionMode = null;  // null, "draw", "challenge"
let maxSelect = 0;
let peekCards = [];

// --- WebSocket ---

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/remote`);

    ws.onopen = () => {
        document.getElementById("connStatus").textContent = "Connected";
        document.getElementById("connStatus").className = "status-badge connected";
    };

    ws.onclose = () => {
        document.getElementById("connStatus").textContent = "Disconnected";
        document.getElementById("connStatus").className = "status-badge disconnected";
        setTimeout(connect, 2000);
    };

    ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        handleMessage(msg);
    };
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

// --- Message handling ---

function handleMessage(msg) {
    switch (msg.type) {
        case "new_hand":
            hand = [];
            selectedSlots = [];
            actionMode = null;
            peekCards = [];
            document.getElementById("gameTitle").textContent = msg.game_name;
            if (msg.wild_label) {
                document.getElementById("wildDisplay").style.display = "";
                document.getElementById("wildDisplay").textContent = msg.wild_label;
            } else {
                document.getElementById("wildDisplay").style.display = "none";
            }
            updateHand();
            hideAction();
            hidePeek();
            break;

        case "card_dealt":
            hand.push({
                slot_number: msg.slot_number,
                card: msg.card,
                card_type: msg.card_type,
                status: "active",
            });
            updateHand();
            break;

        case "draw_prompt":
            actionMode = "draw";
            maxSelect = msg.max_draw;
            selectedSlots = [];
            showAction(`Draw round ${msg.draw_round}: Select up to ${msg.max_draw} cards to discard`);
            updateHand();
            break;

        case "challenge_prompt":
            actionMode = "challenge";
            maxSelect = msg.select_cards;
            selectedSlots = [];
            showAction(msg.label + ` (select ${msg.select_cards} cards)`, true);
            updateHand();
            break;

        case "discard_acknowledged":
            msg.slot_numbers.forEach(num => {
                const slot = hand.find(s => s.slot_number === num);
                if (slot) slot.status = "discarded";
            });
            actionMode = null;
            selectedSlots = [];
            hideAction();
            updateHand();
            break;

        case "challenge_acknowledged":
            msg.slot_numbers.forEach(num => {
                const slot = hand.find(s => s.slot_number === num);
                if (slot) slot.status = "challenged";
            });
            actionMode = null;
            selectedSlots = [];
            hideAction();
            updateHand();
            break;

        case "pass_challenge_acknowledged":
            actionMode = null;
            selectedSlots = [];
            hideAction();
            break;

        case "wild_card_update":
            document.getElementById("wildDisplay").style.display = "";
            document.getElementById("wildDisplay").textContent = msg.label;
            break;

        case "peek_card":
            peekCards.push({ card: msg.card, label: msg.label });
            updatePeek();
            break;

        case "hand_over":
            actionMode = null;
            selectedSlots = [];
            hideAction();
            document.getElementById("gameTitle").textContent = "Hand over";
            break;
    }
}

// --- UI ---

function updateHand() {
    const grid = document.getElementById("handGrid");
    const activeCards = hand.filter(s => s.status === "active" || s.status === "challenged");
    const discardedCards = hand.filter(s => s.status === "discarded");

    if (hand.length === 0) {
        grid.innerHTML = '<span style="color:#666; font-style:italic">No cards yet</span>';
        return;
    }

    let html = "";

    // Active cards first
    activeCards.forEach(slot => {
        html += cardHTML(slot);
    });

    // Discarded cards (greyed out)
    discardedCards.forEach(slot => {
        html += cardHTML(slot);
    });

    grid.innerHTML = html;

    // Add click handlers if in selection mode
    if (actionMode) {
        document.querySelectorAll(".card:not(.discarded):not(.challenged)").forEach(el => {
            el.addEventListener("click", () => toggleSelect(parseInt(el.dataset.slot)));
        });
    }
}

function cardHTML(slot) {
    const card = slot.card;
    const isRed = card.suit === "hearts" || card.suit === "diamonds";
    const colorClass = isRed ? "red" : "black";
    const typeClass = slot.card_type;
    const statusClass = slot.status !== "active" ? slot.status : "";
    const isSelected = selectedSlots.includes(slot.slot_number);

    return `<div class="card ${typeClass} ${colorClass} ${statusClass} ${isSelected ? 'selected' : ''}"
                 data-slot="${slot.slot_number}"
                 style="${actionMode && slot.status === 'active' ? 'cursor:pointer' : ''}">
        <span class="slot-num">${slot.slot_number}</span>
        <span class="rank">${card.rank}</span>
        <span class="suit">${SUIT_SYMBOLS[card.suit] || card.suit}</span>
        <span class="card-type-badge">${slot.card_type}</span>
    </div>`;
}

function toggleSelect(slotNum) {
    if (!actionMode) return;

    const idx = selectedSlots.indexOf(slotNum);
    if (idx >= 0) {
        selectedSlots.splice(idx, 1);
    } else {
        if (selectedSlots.length < maxSelect) {
            selectedSlots.push(slotNum);
        }
    }

    updateHand();

    // Update submit button
    const btn = document.getElementById("actionBtn");
    if (actionMode === "draw") {
        btn.disabled = false; // Can submit 0 discards (stand pat)
        btn.textContent = selectedSlots.length > 0
            ? `Discard ${selectedSlots.length} card(s)`
            : "Stand Pat (keep all)";
    } else if (actionMode === "challenge") {
        btn.disabled = selectedSlots.length !== maxSelect;
        btn.textContent = `Challenge with ${selectedSlots.length}/${maxSelect} cards`;
    }
}

function showAction(text, showPass = false) {
    const area = document.getElementById("actionArea");
    area.style.display = "";
    document.getElementById("actionPrompt").textContent = text;
    document.getElementById("passBtn").style.display = showPass ? "" : "none";

    const btn = document.getElementById("actionBtn");
    if (actionMode === "draw") {
        btn.textContent = "Stand Pat (keep all)";
        btn.disabled = false;
    } else {
        btn.textContent = "Submit";
        btn.disabled = true;
    }
}

function hideAction() {
    document.getElementById("actionArea").style.display = "none";
}

function submitAction() {
    if (actionMode === "draw") {
        send({ type: "discard_request", slot_numbers: selectedSlots });
    } else if (actionMode === "challenge") {
        send({ type: "challenge_request", slot_numbers: selectedSlots });
    }
}

function passChallenge() {
    send({ type: "pass_challenge" });
}

function updatePeek() {
    const section = document.getElementById("peekSection");
    const grid = document.getElementById("peekGrid");

    if (peekCards.length === 0) {
        section.style.display = "none";
        return;
    }

    section.style.display = "";
    grid.innerHTML = peekCards.map(p => {
        const card = p.card;
        const isRed = card.suit === "hearts" || card.suit === "diamonds";
        const colorClass = isRed ? "red" : "black";
        return `<div class="card up ${colorClass}">
            <span class="rank">${card.rank}</span>
            <span class="suit">${SUIT_SYMBOLS[card.suit]}</span>
        </div>`;
    }).join("");
}

function hidePeek() {
    document.getElementById("peekSection").style.display = "none";
    document.getElementById("peekGrid").innerHTML = "";
}

// --- Init ---

connect();
