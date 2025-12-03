"use client";

import type { HarnessType } from "@/types";

interface Props {
  value: HarnessType;
  onChange: (harness: HarnessType) => void;
  detectedFormat?: "harbor" | "terminus" | "unknown";
  disabled?: boolean;
}

export function HarnessSelector({
  value,
  onChange,
  detectedFormat,
  disabled,
}: Props) {
  const options: { value: HarnessType; label: string; description: string }[] =
    [
      {
        value: "harbor",
        label: "Harbor",
        description: "Uses task.toml configuration",
      },
      {
        value: "terminus",
        label: "Terminus (Terminal-Bench)",
        description: "Uses task.yaml configuration",
      },
    ];

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Select Harness
      </label>
      {detectedFormat && detectedFormat !== "unknown" && (
        <p className="text-sm text-blue-600">
          Detected format:{" "}
          {detectedFormat === "harbor"
            ? "task.toml (Harbor)"
            : "task.yaml (Terminus)"}
        </p>
      )}
      <div className="grid grid-cols-2 gap-4">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(option.value)}
            className={`
              p-4 rounded-lg border-2 text-left transition-colors
              ${
                value === option.value
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 hover:border-gray-300"
              }
              ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
            `}
          >
            <div className="font-medium text-gray-900">{option.label}</div>
            <div className="text-sm text-gray-500 mt-1">
              {option.description}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
