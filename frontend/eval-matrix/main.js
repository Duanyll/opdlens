import { loadRunConfigs, loadRuns, loadScalarLogs } from "./api.js";
import { numericMetrics } from "./data.js";
import {
  currentProject,
  ensureNavButton,
  isEvalPath,
  pageContent,
  selectedRunNames,
  sidebarRunColors,
} from "./trackio-dom.js";
import { bindControls, controls, renderControls, renderTable, shell } from "./view.js";

const state = {
  project: "",
  allRuns: [],
  selectedRuns: [],
  runLogs: [],
  runConfigs: {},
  colors: new Map(),
  renderKey: "",
  loadingKey: "",
};

let scheduleTimer = null;

function ensureHost(content) {
  content.classList.add("op-eval-active");
  let host = content.querySelector(":scope > .op-eval-root");
  if (!host) {
    host = document.createElement("div");
    host.className = "op-eval-root";
    content.appendChild(host);
  }
  return host;
}

function teardown() {
  document.querySelectorAll(".page-content.op-eval-active").forEach((node) => node.classList.remove("op-eval-active"));
  document.querySelectorAll(".op-eval-root").forEach((node) => node.remove());
  state.renderKey = "";
  state.loadingKey = "";
}

function selectionKey(project, names) {
  return JSON.stringify([project, controls.scope, controls.scope === "all" ? [] : names]);
}

async function renderIfNeeded({ force = false } = {}) {
  ensureNavButton();
  if (!isEvalPath()) {
    teardown();
    return;
  }
  const content = pageContent();
  const project = currentProject();
  if (!content || !project) return;
  const names = selectedRunNames();
  const key = selectionKey(project, names);
  if (!force && state.renderKey === key && content.querySelector(":scope > .op-eval-root")) return;
  if (state.loadingKey === key) return;

  state.loadingKey = key;
  const host = ensureHost(content);
  host.innerHTML = shell(project);
  try {
    const [allRuns, runConfigs] = await Promise.all([loadRuns(project), loadRunConfigs(project)]);
    const nameSet = new Set(names);
    const selectedRuns = allRuns.filter((run) => nameSet.has(run.name));
    const runsToLoad = controls.scope === "all" ? allRuns : selectedRuns;
    const runLogs = await loadScalarLogs(project, runsToLoad);
    state.project = project;
    state.allRuns = allRuns;
    state.selectedRuns = selectedRuns;
    state.runLogs = runLogs;
    state.runConfigs = runConfigs || {};
    state.colors = sidebarRunColors();

    renderControls(host, numericMetrics(runLogs));
    bindControls(host, state, {
      reload: () => renderIfNeeded({ force: true }),
      refresh: () => renderIfNeeded({ force: true }),
    });
    renderTable(host, state);
    state.renderKey = key;
  } catch (error) {
    console.error("opdlens Eval Matrix failed:", error);
    host.querySelector(".op-eval-subtitle").textContent = "Could not load Trackio eval data.";
    host.querySelector(".op-eval-table-wrap").innerHTML = `<div class="op-eval-empty op-eval-error">${String(error.message || error)}</div>`;
  } finally {
    state.loadingKey = "";
  }
}

function schedule(delay = 100) {
  window.clearTimeout(scheduleTimer);
  scheduleTimer = window.setTimeout(() => renderIfNeeded(), delay);
}

function install() {
  const observer = new MutationObserver(() => schedule());
  observer.observe(document.body, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["checked", "value", "class"],
  });
  window.addEventListener("popstate", () => schedule(0));
  document.addEventListener("change", () => schedule(0), true);
  schedule(300);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", install, { once: true });
} else {
  install();
}
