function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) {
    node.className = cls;
  }
  if (text !== undefined) {
    node.textContent = text;
  }
  return node;
}

function formatValue(spec, value) {
  if (spec.format) {
    return spec.format(value);
  }
  if (spec.step && spec.step < 1) {
    const digits = Math.max(0, Math.ceil(-Math.log10(spec.step)));
    return value.toFixed(digits);
  }
  return String(value);
}

export function buildControls(root, schema, config, onChange) {
  const inputs = new Map();

  for (const group of schema) {
    const section = el('section', 'group');
    section.appendChild(el('h2', null, group.title));

    for (const spec of group.items) {
      const row = el('div', `row row-${spec.type}`);

      if (spec.type === 'range') {
        const head = el('div', 'row-head');
        head.appendChild(el('label', null, spec.label));
        const readout = el('span', 'readout', formatValue(spec, config[spec.key]));
        head.appendChild(readout);
        row.appendChild(head);

        const input = el('input');
        input.type = 'range';
        input.min = spec.min;
        input.max = spec.max;
        input.step = spec.step;
        input.value = spec.unmap ? spec.unmap(config[spec.key]) : config[spec.key];
        input.addEventListener('input', () => {
          const raw = Number(input.value);
          config[spec.key] = spec.map ? spec.map(raw) : raw;
          readout.textContent = formatValue(spec, config[spec.key]);
          onChange(spec.key);
        });
        row.appendChild(input);
        inputs.set(spec.key, { input, readout, spec });
      } else if (spec.type === 'select') {
        row.appendChild(el('label', null, spec.label));
        const input = el('select');
        for (const option of spec.options) {
          const isPair = option !== null && typeof option === 'object';
          const opt = el('option', null, isPair ? option.label : String(option));
          opt.value = String(isPair ? option.value : option);
          input.appendChild(opt);
        }
        input.value = String(config[spec.key]);
        input.addEventListener('change', () => {
          const raw = input.value;
          config[spec.key] = spec.numeric ? Number(raw) : raw;
          onChange(spec.key);
        });
        row.appendChild(input);
        inputs.set(spec.key, { input, spec });
      } else if (spec.type === 'toggle') {
        const label = el('label', 'toggle');
        const input = el('input');
        input.type = 'checkbox';
        input.checked = Boolean(config[spec.key]);
        input.addEventListener('change', () => {
          config[spec.key] = input.checked;
          onChange(spec.key);
        });
        label.appendChild(input);
        label.appendChild(el('span', null, spec.label));
        row.appendChild(label);
        inputs.set(spec.key, { input, spec });
      } else if (spec.type === 'button') {
        const button = el('button', 'action', spec.label);
        button.addEventListener('click', () => onChange(spec.key));
        row.appendChild(button);
      }

      section.appendChild(row);
    }
    root.appendChild(section);
  }

  return {
    sync() {
      for (const [key, entry] of inputs) {
        if (entry.spec.type === 'toggle') {
          entry.input.checked = Boolean(config[key]);
        } else if (entry.spec.unmap) {
          entry.input.value = String(entry.spec.unmap(config[key]));
        } else {
          entry.input.value = String(config[key]);
        }
        if (entry.readout) {
          entry.readout.textContent = formatValue(entry.spec, config[key]);
        }
      }
    },
  };
}
