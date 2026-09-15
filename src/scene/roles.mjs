/**
 * The application's own vocabulary for labelled fields.
 *
 * Datasets disagree about what to call things. ScanNet's Pointcept export has
 * `segment20`, `segment200` and `instance`; S3DIS has `segment` and `instance`;
 * a PCD might have `label` and `object_id`; somebody else ships `sem_seg` and
 * `cluster_id`. Downstream, none of that should matter -- the class legend, the
 * object library, the inference payload and the interpretability tools all want
 * the same two things:
 *
 *   semantic   what kind of thing a point is        (a handful of classes)
 *   instance   which particular object it belongs to (one value per object)
 *
 * So each dataset declares its own nomenclature once, and everything after the
 * reader speaks in roles instead of field names:
 *
 *   "roles": { "semantic": "segment", "instance": "instance" }
 *
 * in a dataset's entry in `config/app.json`, or in a `classes.json` -- per
 * dataset, per folder or per scene, with the closer file winning. A field keeps
 * its own name everywhere it is shown; only the *role* is translated.
 *
 * The mapping is optional. Without one the roles are inferred, which is what
 * makes an undeclared folder of scenes work at all -- but inference is a guess,
 * and a dataset that says what it means never has to be guessed at.
 */

/** The roles, in the order they are resolved. Semantic first: it constrains instance. */
export const ROLES = ["semantic", "instance"];

/**
 * Field names that conventionally mean each role, best first. Matched against
 * the field's name in the source file, lowercased.
 */
export const CONVENTION = {
  semantic: [
    "classification", "label", "labels", "semantic", "semantics", "sem_seg",
    "segmentation", "segment", "seg", "class", "category", "panoptic",
    // Datasets shipping several label sets: prefer the coarse one, which is the
    // more useful default to open on.
    "segment20", "segment200",
  ],
  instance: [
    "instance", "instances", "instance_id", "object_id", "obj_id", "object",
    "objects", "cluster", "cluster_id", "segment_id", "track_id", "part_id",
  ],
};

/** Names that mean "an object id", for the guard in inference. */
export const INSTANCE_NAMES = CONVENTION.instance;

/**
 * Reads a role map out of a config object, accepting the older spellings.
 *
 * `primaryField` predates roles and meant exactly `roles.semantic`;
 * `instanceField` is the same idea for the library. Both still work.
 */
export function roleMapFrom(cfg) {
  if (!cfg) return {};
  const out = {};
  for (const role of ROLES) {
    const declared = cfg.roles?.[role];
    if (typeof declared === "string" && declared) out[role] = declared;
  }
  if (!out.semantic && typeof cfg.primaryField === "string" && cfg.primaryField) {
    out.semantic = cfg.primaryField;
  }
  if (!out.instance && typeof cfg.instanceField === "string" && cfg.instanceField) {
    out.instance = cfg.instanceField;
  }
  return out;
}

/**
 * Merges role maps, later sources winning per role. Used to layer a dataset's
 * declaration under a folder's under a scene's under an explicit request.
 */
export function mergeRoleMaps(...maps) {
  const out = {};
  for (const map of maps) {
    for (const role of ROLES) if (map?.[role]) out[role] = map[role];
  }
  return out;
}

const named = (candidates, wanted) =>
  candidates.find((s) => s.source === wanted || s.name === wanted) ?? null;

/**
 * Resolves the roles against a scene's categorical fields.
 *
 * Per role, in order:
 *   1. the declared mapping -- the dataset said so,
 *   2. a conventional name,
 *   3. inference from the data.
 *
 * Inference never reads field order, which is the one thing that carries no
 * meaning: on S3DIS (instance.npy sorting before segment.npy) taking the first
 * categorical field handed the class legend the object ids and the object
 * library the semantic classes, the two roles exactly swapped. What it reads
 * instead is cardinality. A semantic field has a handful of classes; an instance
 * field has one value per object -- so the fewest distinct values is the
 * semantic one and the most is the instance one, and a field named like an id is
 * never taken as semantic while anything else is available.
 *
 * @param categorical  described.categorical
 * @param declared     a role map, from roleMapFrom()/mergeRoleMaps()
 * @returns {{ semantic, instance, resolvedBy: {semantic, instance} }}
 *          field entries (or null), and how each one was arrived at.
 */
export function resolveRoles(categorical, declared = {}) {
  const out = { semantic: null, instance: null, resolvedBy: { semantic: null, instance: null } };
  if (!categorical?.length) return out;

  for (const role of ROLES) {
    // A field cannot hold both roles at once.
    const taken = new Set(ROLES.map((r) => out[r]?.name).filter(Boolean));
    const pool = categorical.filter((s) => !taken.has(s.name));
    if (pool.length === 0) continue;

    if (declared[role]) {
      const hit = named(pool, declared[role]);
      if (hit) { out[role] = hit; out.resolvedBy[role] = "declared"; continue; }
      // Declared but not in this scene. Two different situations reach here and
      // both are handled by carrying on: a scene that genuinely has no such
      // field (ScanNet's test split withholds the labels) finds nothing below
      // either and ends up empty, which is the truth; a scene that simply is not
      // what the dataset describes -- a foreign folder dropped inside a broad
      // root like `scenes/` -- gets read on its own terms instead of inheriting
      // a schema that was never about it.
    }

    const byName = CONVENTION[role]
      .map((want) => pool.find((s) => s.source.toLowerCase() === want))
      .find(Boolean);
    if (byName) {
      out[role] = byName;
      out.resolvedBy[role] = declared[role] ? "name-after-declared-missing" : "name";
      continue;
    }

    // An id-shaped name is never the semantic field, even as the last candidate
    // standing. A cloud whose only label is `instance` has no classes to show --
    // it has objects -- and saying so leaves the object library working and the
    // field still colourable as a plain categorical attribute. Calling it the
    // class field instead would produce a legend of one entry per object and an
    // empty library, which is the failure this whole module exists to prevent.
    const idish = (f) => INSTANCE_NAMES.includes(f.source.toLowerCase());
    if (role === "semantic") {
      out.semantic = pool.filter((f) => !idish(f))
        .sort((a, b) => a.numClasses - b.numClasses)[0] ?? null;
    } else {
      out.instance = pool.slice().sort((a, b) => b.numClasses - a.numClasses)[0] ?? null;
    }
    if (out[role]) out.resolvedBy[role] = declared[role] ? "inferred-after-declared-missing" : "inferred";
    else if (declared[role]) out.resolvedBy[role] = "declared-missing";
  }
  return out;
}

/** The part of a resolution worth recording in scene.json. */
export const describeRoles = (roles) => ({
  semantic: roles.semantic
    ? { field: roles.semantic.name, source: roles.semantic.source, classes: roles.semantic.numClasses }
    : null,
  instance: roles.instance
    ? { field: roles.instance.name, source: roles.instance.source, classes: roles.instance.numClasses }
    : null,
  resolvedBy: roles.resolvedBy,
});
