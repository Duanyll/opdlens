export function isEvalPath() {
  const base = window.__trackio_base || "";
  const raw = base && window.location.pathname.startsWith(base)
    ? window.location.pathname.slice(base.length)
    : window.location.pathname;
  return (raw.replace(/\/+$/, "") || "/") === "/evals";
}

export function currentProject() {
  const locked = document.querySelector(".locked-project");
  if (locked?.textContent?.trim() && locked.textContent.trim() !== "—") {
    return locked.textContent.trim();
  }
  return document.querySelector('input[aria-label="Project"]')?.value?.trim() || "";
}

export function selectedRunNames() {
  const names = [];
  document.querySelectorAll(".sidebar .checkbox-item").forEach((label) => {
    const input = label.querySelector('input[type="checkbox"]');
    const name = label.querySelector(".run-name")?.getAttribute("title")?.trim();
    if (input?.checked && name) names.push(name);
  });
  return [...new Set(names)];
}

export function sidebarRunColors() {
  const colors = new Map();
  document.querySelectorAll(".sidebar .checkbox-item").forEach((label) => {
    const name = label.querySelector(".run-name")?.getAttribute("title")?.trim();
    const dot = label.querySelector(".color-dot");
    if (!name || !dot) return;
    const color = dot.style.backgroundColor || window.getComputedStyle(dot).backgroundColor;
    if (color) colors.set(name, color);
  });
  return colors;
}

export function pageContent() {
  return document.querySelector(".page-content");
}

export function navigateToEvals() {
  const base = window.__trackio_base || "";
  const query = window.location.search;
  window.history.pushState({}, "", `${base}/evals${query}`);
  window.dispatchEvent(new PopStateEvent("popstate"));
}

export function ensureNavButton() {
  const tabs = document.querySelector(".navbar .nav-tabs");
  if (!tabs) return;
  let button = tabs.querySelector(".op-eval-nav");
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.className = "nav-link op-eval-nav";
    button.textContent = "Eval Matrix";
    button.title = "Compare eval results by run and step";
    button.addEventListener("click", navigateToEvals);
    tabs.querySelector(".settings-btn")?.before(button);
  }
  if (isEvalPath()) {
    tabs.querySelectorAll(".nav-link.active").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
  } else {
    button.classList.remove("active");
  }
}
