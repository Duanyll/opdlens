import test from "node:test";
import assert from "node:assert/strict";

import {
  buildMatrix,
  columnRanges,
  columnSortDirection,
  defaultMetric,
  filterAndSortRows,
  filterSteps,
  heatLevel,
  numericMetrics,
  nextColumnSort,
  parseStepFilter,
} from "./data.js";

const runs = [
  { id: "a", name: "run-a" },
  { id: "b", name: "run-b" },
];
const logs = [
  { run_id: "a", logs: [{ step: 0, "eval/test/acc": 0.5 }, { step: 50, "eval/test/acc": 0.7 }] },
  { run_id: "b", logs: [{ step: 0, "eval/test/acc": 0.6 }, { step: 100, "eval/test/acc": 0.8 }] },
];

test("discovers eval metrics and selects accuracy", () => {
  const metrics = numericMetrics(logs);
  assert.deepEqual(metrics, ["eval/test/acc"]);
  assert.equal(defaultMetric(metrics), "eval/test/acc");
});

test("builds run rows over the union of eval steps", () => {
  const matrix = buildMatrix(logs, runs, "eval/test/acc");
  assert.deepEqual(matrix.steps, [0, 50, 100]);
  assert.equal(matrix.rows[0].latestValue, 0.7);
  assert.ok(Math.abs(matrix.rows[1].delta - 0.2) < 1e-12);
});

test("parses exact, range, and stride step filters", () => {
  const result = filterSteps([0, 25, 50, 75, 100, 150], "0,50:25:100,150");
  assert.equal(result.valid, true);
  assert.deepEqual(result.steps, [0, 50, 75, 100, 150]);
  assert.equal(parseStepFilter("0:nope:100").valid, false);
});

test("filters names by terms and sorts by latest value", () => {
  const rows = buildMatrix(logs, runs, "eval/test/acc").rows;
  assert.equal(filterAndSortRows(rows, "run b", "latest_desc")[0].name, "run-b");
  assert.deepEqual(filterAndSortRows(rows, "", "latest_desc").map((row) => row.name), ["run-b", "run-a"]);
});

test("sorts by an individual step and toggles column direction", () => {
  const rows = buildMatrix(logs, runs, "eval/test/acc").rows;
  assert.deepEqual(filterAndSortRows(rows, "", "value:0:desc").map((row) => row.name), ["run-b", "run-a"]);
  assert.equal(nextColumnSort("name_asc", "name"), "name_desc");
  assert.equal(nextColumnSort("name_desc", "value:50"), "value:50_desc");
  assert.equal(nextColumnSort("value:50_desc", "value:50"), "value:50_asc");
  assert.equal(columnSortDirection("value:50_asc", "value:50"), "asc");
});

test("heat levels can prefer high or low values", () => {
  const range = { low: 0.4, center: 0.6, high: 0.8 };
  assert.equal(heatLevel(0.8, range, "higher"), 1);
  assert.equal(heatLevel(0.8, range, "lower"), 0);
  assert.equal(heatLevel(0.6, range, "higher"), 0.5);
  assert.equal(heatLevel(0.8, range, "off"), null);
});

test("robust color ranges ignore isolated outliers", () => {
  const values = [0.1, 0.8, 0.81, 0.82, 0.83, 0.84, 0.99];
  const rows = values.map((value) => ({ points: new Map([[50, { value }]]) }));
  const range = columnRanges(rows, [50]).get(50);
  assert.deepEqual(range, { low: 0.8, center: 0.82, high: 0.84 });
  assert.equal(heatLevel(0.1, range, "higher"), 0);
  assert.ok(heatLevel(0.83, range, "higher") > 0.5);
});
