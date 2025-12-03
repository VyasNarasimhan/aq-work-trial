"use client";

import type { ModelType } from "@/types";
import { MODEL_OPTIONS, DEFAULT_MODEL } from "@/types";

interface Props {
  value: ModelType;
  onChange: (model: ModelType) => void;
  disabled?: boolean;
}

export function ModelSelector({ value, onChange, disabled }: Props) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">
        Select Model
      </label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as ModelType)}
        disabled={disabled}
        className={`
          w-full p-3 rounded-lg border-2 border-gray-200
          bg-white text-gray-900
          focus:border-blue-500 focus:ring-2 focus:ring-blue-200 focus:outline-none
          transition-colors
          ${disabled ? "opacity-50 cursor-not-allowed bg-gray-50" : "cursor-pointer hover:border-gray-300"}
        `}
      >
        {MODEL_OPTIONS.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}
