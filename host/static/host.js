// Host (dealer) control panel WebSocket client

const SUIT_SYMBOLS = {
    hearts: "\u2665", diamonds: "\u2666",
    clubs: "\u2663", spades: "\u2660"
};

let ws = null;
let state = null;

// --- WebSocket ---

function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}/ws/host`);

    ws.onopen = () => {
        document.getElementById("remoteStatus").textContent = "WS: Connected";
        document.getElementById("remoteStatus").className = "status-badge connected";
    };

    ws.onclose = () => {
        document.getElementById("remoteStatus").textContent = "WS: Disconnected";
        document.getElementById("remoteStatus").className = "status-badge disconnected";
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
    if (msg.type === "hand_update") {
        state = msg.state;
        updateUI();
    } else if (msg.type === "scan_confirm_needed") {
        showScanConfirm(msg);
    } else if (msg.type === "pi_error") {
        showToast("Pi Error: " + msg.message, "error");
    }
}

// --- UI Updates ---

function updateUI() {
    if (!state) return;

    // Phase display
    const phaseSection = document.getElementById("phaseSection");
    const phaseDisplay = document.getElementById("phaseDisplay");
    const continueBtn = document.getElementById("continueBtn");
    const wildDisplay = document.getElementById("wildDisplay");

    if (state.game_name) {
        phaseSection.style.display = "";
        phaseDisplay.textContent = state.current_phase;

        // Set phase class
        phaseDisplay.className = "phase-indicator " + state.state;

        // Continue button for betting
        continueBtn.style.display = state.state === "betting" ? "" : "none";

        // Wild card display
        if (state.wild_label) {
            wildDisplay.style.display = "";
            wildDisplay.textContent = state.wild_label;
        } else {
            wildDisplay.style.display = "none";
        }
    } else {
        phaseSection.style.display = "none";
    }

    // Hand display
    const grid = document.getElementById("handGrid");
    if (state.slots && state.slots.length > 0) {
        grid.innerHTML = state.slots.map(slot => cardHTML(slot)).join("");
    } else if (state.game_name) {
        grid.innerHTML = '<span style="color:#666; font-style:italic">Waiting for first card...</span>';
    } else {
        grid.innerHTML = '<span style="color:#666; font-style:italic">No cards dealt yet</span>';
    }
}

function cardHTML(slot) {
    const card = slot.card;
    const isRed = card.suit === "hearts" || card.suit === "diamonds";
    const colorClass = isRed ? "red" : "black";
    const typeClass = slot.card_type;
    const statusClass = slot.status !== "active" ? slot.status : "";

    return `<div class="card ${typeClass} ${colorClass} ${statusClass}">
        <span class="slot-num">${slot.slot_number}</span>
        <span class="rank">${card.rank}</span>
        <span class="suit">${SUIT_SYMBOLS[card.suit] || card.suit}</span>
        <span class="card-type-badge">${slot.card_type}</span>
    </div>`;
}

// --- Actions ---

function newHand() {
    const game = document.getElementById("gameSelect").value;
    if (!game) return;
    send({ type: "new_hand", game_name: game });
}

function endHand() {
    send({ type: "end_hand" });
}

function continueBetting() {
    send({ type: "continue_betting" });
}

function simulateScan() {
    const rank = document.getElementById("simRank").value;
    const suit = document.getElementById("simSuit").value;
    send({ type: "simulate_scan", card: { rank, suit } });
}

function peekScan() {
    const rank = document.getElementById("simRank").value;
    const suit = document.getElementById("simSuit").value;
    send({ type: "peek_card", card: { rank, suit }, label: "Peek" });
}

function showScanConfirm(msg) {
    // For now, auto-accept. TODO: show confirmation dialog
    send({ type: "scan_confirm", card: msg.card });
}

function showToast(message, type = "") {
    const toast = document.createElement("div");
    toast.className = "toast " + type;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// --- Init ---

async function init() {
    // Load game list
    const resp = await fetch("/api/games");
    const data = await resp.json();
    const select = document.getElementById("gameSelect");
    data.games.forEach(name => {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    });

    // Check Pi status
    const piResp = await fetch("/api/pi/status");
    const piData = await piResp.json();
    const piBadge = document.getElementById("piStatus");
    if (piData.connected) {
        piBadge.textContent = piData.mock ? "Pi: Mock" : "Pi: Connected";
        piBadge.className = "status-badge connected";
    }

    connect();
}

init();
