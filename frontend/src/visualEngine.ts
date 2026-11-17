// Pure helpers of the Visual engine dialog (VisualEngineControl.tsx), kept apart so node tests can import them.

export type VisualRole = { role: string; label: string; default: string }
export type VisualModel = { key: string; label: string; model_id: string; roles: string[]; note: string; hears_audio: boolean; install: string | null; available: boolean; detail: string }
export type VisualModels = { roles: VisualRole[]; models: VisualModel[]; selected: Record<string, string>; worker_reported: boolean }

/** The models that can run a role, the default first. A missing model stays listed but cannot be chosen,
 *  unless it is the current choice (so the select never shows a blank). */
export function roleOptions(data: VisualModels, role: VisualRole) {
  return data.models
    .filter(model => model.roles.includes(role.role))
    .sort((a, b) => Number(b.key === role.default) - Number(a.key === role.default))
    .map(model => ({
      key: model.key,
      label: model.label + (model.key === role.default ? ' (default)' : '') + (model.available ? '' : ` · ${model.detail}`),
      disabled: !model.available && model.key !== role.default && data.selected[role.role] !== model.key,
    }))
}

/** Short summary for the header button: "Default" or the non-default models in use. */
export function engineBadge(data: VisualModels | null) {
  if (!data) return 'checking…'
  const custom = data.roles.filter(role => data.selected[role.role] && data.selected[role.role] !== role.default)
  if (!custom.length) return 'Default'
  const names = [...new Set(custom.map(role => data.models.find(model => model.key === data.selected[role.role])?.label.split(' · ')[0] ?? data.selected[role.role]))]
  return names.join(' + ')
}

/** Models that are not installed yet, with the command that installs them. */
export function installHints(data: VisualModels) {
  return data.models.filter(model => !model.available && model.install).map(model => ({ key: model.key, label: model.label, detail: model.detail, command: model.install as string }))
}
