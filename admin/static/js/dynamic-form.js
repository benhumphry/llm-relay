/**
 * Dynamic Form Component for LLM Relay Admin UI
 *
 * Renders form fields dynamically based on plugin field definitions.
 * Used by document stores, live data sources, and other plugin-based forms.
 *
 * Usage in Alpine.js:
 *
 * <div x-data="dynamicForm({
 *     fieldsUrl: '/admin/api/document-sources/{source_type}/fields',
 *     config: form.config,
 *     oauthAccounts: oauthAccounts
 * })">
 *     <template x-for="field in fields" :key="field.name">
 *         <div x-html="renderField(field)"></div>
 *     </template>
 * </div>
 */

/**
 * Alpine.js component factory for dynamic forms
 */
function dynamicForm(options = {}) {
    return {
        fields: [],
        config: options.config || {},
        oauthAccounts: options.oauthAccounts || { google: [], microsoft: [] },
        loading: false,
        error: null,

        /**
         * Load field definitions from API
         */
        async loadFields(sourceType) {
            if (!sourceType) {
                this.fields = [];
                return;
            }

            this.loading = true;
            this.error = null;

            try {
                const url = options.fieldsUrl.replace('{source_type}', sourceType);
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error(`Failed to load fields: ${response.statusText}`);
                }

                const data = await response.json();
                this.fields = data.fields || [];

                // Initialize config with defaults for new fields
                for (const field of this.fields) {
                    if (this.config[field.name] === undefined && field.default !== undefined) {
                        this.config[field.name] = field.default;
                    }
                }
            } catch (err) {
                console.error('Failed to load fields:', err);
                this.error = err.message;
                this.fields = [];
            } finally {
                this.loading = false;
            }
        },

        /**
         * Check if a field should be visible based on depends_on conditions
         */
        isFieldVisible(field) {
            if (!field.depends_on || Object.keys(field.depends_on).length === 0) {
                return true;
            }

            for (const [fieldName, expectedValue] of Object.entries(field.depends_on)) {
                if (this.config[fieldName] !== expectedValue) {
                    return false;
                }
            }
            return true;
        },

        /**
         * Check if a field should be hidden due to env var being set
         */
        isEnvVarSet(field) {
            return field.env_var_set === true;
        },

        /**
         * Get the appropriate input element for a field type
         */
        getInputType(fieldType) {
            const typeMap = {
                'text': 'text',
                'password': 'password',
                'integer': 'number',
                'number': 'number',
                'textarea': 'textarea',
                'boolean': 'checkbox',
                'select': 'select',
                'multiselect': 'select',
                'oauth_account': 'oauth_account',
                'folder_picker': 'folder_picker',
                'calendar_picker': 'calendar_picker',
                'channel_picker': 'channel_picker',
                'label_picker': 'label_picker',
                'tasklist_picker': 'tasklist_picker',
            };
            return typeMap[fieldType] || 'text';
        },

        /**
         * Get OAuth accounts for a provider
         */
        getOAuthAccounts(provider) {
            if (!provider || !this.oauthAccounts) return [];
            return this.oauthAccounts[provider] || [];
        },

        /**
         * Render a field (returns HTML string for x-html binding)
         * Note: This is a simplified renderer. Complex fields like pickers
         * should be implemented as separate Alpine components.
         */
        renderField(field) {
            if (this.isEnvVarSet(field)) {
                return `<div class="form-control">
                    <label class="label">
                        <span class="label-text">${field.label}</span>
                        <span class="badge badge-success badge-sm">Configured via env</span>
                    </label>
                </div>`;
            }

            const inputType = this.getInputType(field.field_type);
            const required = field.required ? 'required' : '';
            const helpText = field.help_text ?
                `<label class="label"><span class="label-text-alt">${field.help_text}</span></label>` : '';

            switch (inputType) {
                case 'checkbox':
                    return `<div class="form-control">
                        <label class="label cursor-pointer justify-start gap-2">
                            <input type="checkbox"
                                x-model="config.${field.name}"
                                class="checkbox checkbox-primary" />
                            <span class="label-text">${field.label}</span>
                        </label>
                        ${helpText}
                    </div>`;

                case 'textarea':
                    return `<div class="form-control">
                        <label class="label">
                            <span class="label-text">${field.label}${field.required ? ' *' : ''}</span>
                        </label>
                        <textarea
                            x-model="config.${field.name}"
                            placeholder="${field.placeholder || ''}"
                            class="textarea textarea-bordered"
                            ${required}
                        ></textarea>
                        ${helpText}
                    </div>`;

                case 'select':
                    const options = (field.options || [])
                        .map(opt => `<option value="${opt.value}">${opt.label}</option>`)
                        .join('');
                    return `<div class="form-control">
                        <label class="label">
                            <span class="label-text">${field.label}${field.required ? ' *' : ''}</span>
                        </label>
                        <select
                            x-model="config.${field.name}"
                            class="select select-bordered"
                            ${required}
                        >
                            <option value="">Select...</option>
                            ${options}
                        </select>
                        ${helpText}
                    </div>`;

                case 'oauth_account':
                    // OAuth account picker - needs special handling
                    const provider = field.picker_options?.provider || 'google';
                    return `<div class="form-control" x-data="{ accounts: getOAuthAccounts('${provider}') }">
                        <label class="label">
                            <span class="label-text">${field.label}${field.required ? ' *' : ''}</span>
                        </label>
                        <div class="flex gap-2">
                            <select
                                x-model="config.${field.name}"
                                class="select select-bordered flex-1"
                                ${required}
                            >
                                <option value="">Select account...</option>
                                <template x-for="acc in accounts" :key="acc.id">
                                    <option :value="acc.id" x-text="acc.email"></option>
                                </template>
                            </select>
                            <button type="button"
                                @click="connectOAuth && connectOAuth('${provider}')"
                                class="btn btn-outline">
                                Connect
                            </button>
                        </div>
                        ${helpText}
                    </div>`;

                default:
                    // Text, password, number inputs
                    const inputAttrs = [];
                    if (field.min_value !== undefined) inputAttrs.push(`min="${field.min_value}"`);
                    if (field.max_value !== undefined) inputAttrs.push(`max="${field.max_value}"`);
                    if (field.min_length !== undefined) inputAttrs.push(`minlength="${field.min_length}"`);
                    if (field.max_length !== undefined) inputAttrs.push(`maxlength="${field.max_length}"`);
                    if (field.pattern) inputAttrs.push(`pattern="${field.pattern}"`);

                    return `<div class="form-control">
                        <label class="label">
                            <span class="label-text">${field.label}${field.required ? ' *' : ''}</span>
                        </label>
                        <input
                            type="${inputType === 'number' ? 'number' : inputType}"
                            x-model="config.${field.name}"
                            placeholder="${field.placeholder || ''}"
                            class="input input-bordered"
                            ${required}
                            ${inputAttrs.join(' ')}
                        />
                        ${helpText}
                    </div>`;
            }
        }
    };
}

// Export for use in templates
window.dynamicForm = dynamicForm;
