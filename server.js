// server.js
require("dotenv").config();
const express = require("express");
const path = require("path");
const fs = require("fs");
const fetch = require("node-fetch"); // npm i node-fetch@2
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json({limit: "1mb"}));

const DATA_DIR = path.join(__dirname, "data");
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

// ---------- helpers ----------
function csvEscape(val) {
  if (val === null || val === undefined) return "";
  const s = String(val);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function appendCSV(rowObj, filePath) {
  const header = Object.keys(rowObj).map(csvEscape).join(",") + "\n";
  const line   = Object.values(rowObj).map(csvEscape).join(",") + "\n";
  const exists = fs.existsSync(filePath);
  if (!exists) fs.writeFileSync(filePath, header, "utf8");
  fs.appendFileSync(filePath, line, "utf8");
}

function pickWebhook(type) {
  switch (type) {
    case "retail":         return process.env.TEAMS_WEBHOOK_URL_Retail;
    case "business":       return process.env.TEAMS_WEBHOOK_URL_Business;
    case "vip":            return process.env.TEAMS_WEBHOOK_URL_VIP;
    case "nonchargeable":  return process.env.TEAMS_WEBHOOK_URL_NC || process.env.TEAMS_WEBHOOK_URL_Monitor;
    case "project":        return process.env.TEAMS_WEBHOOK_URL_Project || process.env.TEAMS_WEBHOOK_URL_Monitor;
    default:               return process.env.TEAMS_WEBHOOK_URL_Monitor;
  }
}

async function sendTeams(type, data) {
  const webhook = pickWebhook(type);
  if (!webhook) return {ok:false, msg:"No webhook set"};

  // Build a simple, tidy card (mirrors your Streamlit style)
  const facts = [];
  const add = (name, v) => { if (v !== undefined && v !== null && String(v).trim() !== "") facts.push({name, value:String(v)}); };

  let title = "Invoice";
  if (type === "retail") {
    title = `📄 Retail — ${data.engineer || ""} — ${data.ticket || ""}`;
    add("Date", data.visit_date, facts);
    add("Stakeholder", data.stakeholder_type, facts);
    add("ASA", data.asa_number, facts);
    add("Store", `${data.store_name || ""} (${data.postcode || ""})`, facts);
  } else if (type === "business") {
    title = `📄 Business — ${data.engineer || ""} — ${data.invoice_type || ""}`;
    add("Date", data.visit_date, facts);
    add("VR Number", data.vr_number, facts);
    add("SLA/Area/Type", data.sla_type || data.area || data.visit_type, facts);
    add("Engineers", data.engineer_count, facts);
  } else if (type === "vip") {
    title = `📄 VIP / Tier 2 — ${data.lead_engineer || ""} — ${data.vr_number || ""}`;
    add("Date", data.visit_date, facts);
    add("Job Type", data.job_type, facts);
  } else if (type === "nonchargeable") {
    title = `📝 Non-chargeable — ${data.engineer || ""} — ${data.activity || ""}`;
    add("Date", data.visit_date, facts);
    add("Oracle Time", data.oracle_hhmm, facts);
  } else if (type === "project") {
    title = `🛠️ Project Install — ${data.engineer || ""}`;
    add("Date", data.install_date, facts);
    add("Venue", data.venue_name, facts);
    add("Postcode", data.venue_postcode, facts);
    add("Hotel Required", data.hotel_required, facts);
  }
  add("Labour",  `£${Number(data.labour_value||0).toFixed(2)}`, facts);
  add("Sub-total",`£${Number(data.sub_total||0).toFixed(2)}`, facts);
  add("VAT",     `£${Number(data.vat||0).toFixed(2)}`, facts);
  add("TOTAL",   `£${Number(data.total_value||0).toFixed(2)}`, facts);

  const card = {
    "@type":"MessageCard","@context":"http://schema.org/extensions",
    "summary": title, "themeColor":"0076D7", "title": title,
    "sections":[{facts, text: (data.notes||"").trim()}]
  };

  const r = await fetch(webhook, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body: JSON.stringify(card)
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return {ok:true, msg:"Sent"};
}

// ---------- API ----------
app.post("/api/submit", async (req,res) => {
  try {
    const { type, data } = req.body || {};
    if (!type || !data) return res.status(400).json({ok:false, error:"Missing type/data"});

    const safe = String(type).toLowerCase().replace(/[^a-z0-9_-]/g,"");
    const file = path.join(DATA_DIR, `${safe}_invoices.csv`);

    appendCSV(data, file);

    let teams = {ok:false, msg:""};
    try { teams = await sendTeams(safe, data); } catch(e){ teams = {ok:false, msg:String(e)}; }

    res.json({ok:true, saved:file, teams});
  } catch (e) {
    res.status(500).json({ok:false, error:String(e)});
  }
});

// serve static site too (optional)
app.use(express.static(__dirname));

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => console.log(`API running on http://localhost:${PORT}`));
