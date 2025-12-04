"use client";

import { useCallback, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { HarnessSelector } from "./HarnessSelector";
import { ModelSelector } from "./ModelSelector";
import type { HarnessType, ModelType } from "@/types";
import { DEFAULT_MODEL } from "@/types";

const DEFAULT_HARNESS: HarnessType = "harbor";

interface UploadResult {
  upload_id: string;
  task_name: string;
  detected_format: "harbor" | "terminus" | "unknown";
}

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<string>("");

  // Harness and model selection state
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [selectedHarness, setSelectedHarness] =
    useState<HarnessType>(DEFAULT_HARNESS);
  const [selectedModel, setSelectedModel] = useState<ModelType>(DEFAULT_MODEL);
  const [startingBenchmark, setStartingBenchmark] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  const handleFile = useCallback(async (file: File) => {
    if (!file.name.endsWith(".zip")) {
      setError("Please upload a .zip file");
      return;
    }

    setUploading(true);
    setError(null);
    setUploadProgress("Uploading file...");

    try {
      const result = await api.uploadFile(file);

      // Show harness selection instead of immediately creating benchmark
      setUploadResult(result);
      // Auto-select harness based on detected format
      if (result.detected_format === "harbor") {
        setSelectedHarness("harbor");
      } else if (result.detected_format === "terminus") {
        setSelectedHarness("terminus");
      }
      setUploading(false);
      setUploadProgress("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setUploading(false);
      setUploadProgress("");
    }
  }, []);

  const handleStartBenchmark = useCallback(async () => {
    if (!uploadResult) return;

    setStartingBenchmark(true);
    setError(null);

    try {
      const { id } = await api.createBenchmark(
        uploadResult.upload_id,
        selectedHarness,
        selectedModel
      );
      router.push(`/benchmarks?id=${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start benchmark");
      setStartingBenchmark(false);
    }
  }, [uploadResult, selectedHarness, selectedModel, router]);

  const handleCancel = useCallback(() => {
    setUploadResult(null);
    setSelectedHarness(DEFAULT_HARNESS);
    setSelectedModel(DEFAULT_MODEL);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);

      const file = e.dataTransfer.files[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFile(file);
      }
    },
    [handleFile]
  );

  const handleClick = () => {
    if (!uploadResult) {
      fileInputRef.current?.click();
    }
  };

  // Show harness selection after upload
  if (uploadResult) {
    return (
      <div className="border-2 border-gray-200 rounded-xl p-8 bg-white">
        <div className="space-y-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Configure Benchmark
            </h3>
            <p className="text-gray-600 mt-1">
              Task: <span className="font-medium">{uploadResult.task_name}</span>
            </p>
          </div>

          <HarnessSelector
            value={selectedHarness}
            onChange={setSelectedHarness}
            detectedFormat={uploadResult.detected_format}
            disabled={startingBenchmark}
          />

          <ModelSelector
            value={selectedModel}
            onChange={setSelectedModel}
            disabled={startingBenchmark}
          />

          {error && (
            <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-red-600 text-sm">{error}</p>
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleCancel}
              disabled={startingBenchmark}
              className="px-4 py-2 text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
            <button
              onClick={handleStartBenchmark}
              disabled={startingBenchmark}
              className="flex-1 px-4 py-2 bg-blue-600 text-white hover:bg-blue-700 rounded-lg transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {startingBenchmark ? (
                <>
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                  Starting...
                </>
              ) : (
                "Start Benchmark"
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={handleClick}
      className={`
        border-2 border-dashed rounded-xl p-12 text-center
        transition-colors cursor-pointer
        ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
        }
        ${uploading ? "pointer-events-none opacity-75" : ""}
      `}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        onChange={handleFileSelect}
        className="hidden"
      />

      {uploading ? (
        <div className="space-y-3">
          <div className="animate-spin h-8 w-8 border-4 border-blue-500 border-t-transparent rounded-full mx-auto" />
          <p className="text-gray-600">{uploadProgress}</p>
        </div>
      ) : (
        <>
          <div className="text-5xl mb-4">📦</div>
          <p className="text-gray-700 font-medium text-lg">
            Drag and drop your task zip file here
          </p>
          <p className="text-gray-500 text-sm mt-2">or click to browse</p>
        </>
      )}

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-600 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}
