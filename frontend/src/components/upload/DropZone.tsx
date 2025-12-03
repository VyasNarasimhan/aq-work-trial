"use client";

import { useCallback, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { HarnessSelector } from "./HarnessSelector";
import { ModelSelector } from "./ModelSelector";
import type { HarnessType, ModelType } from "@/types";
import { DEFAULT_MODEL } from "@/types";

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<string>("");

  // Two-step flow state
  const [uploadResult, setUploadResult] = useState<{
    upload_id: string;
    task_name: string;
    detected_format: "harbor" | "terminus" | "unknown";
  } | null>(null);
  const [selectedHarness, setSelectedHarness] = useState<HarnessType>("harbor");
  const [selectedModel, setSelectedModel] = useState<ModelType>(DEFAULT_MODEL);

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
      setUploadResult({
        upload_id: result.upload_id,
        task_name: result.task_name,
        detected_format: result.detected_format || "unknown",
      });

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

  const handleSubmit = useCallback(async () => {
    if (!uploadResult) return;

    setUploading(true);
    setUploadProgress(
      `Starting ${selectedHarness} benchmark for "${uploadResult.task_name}"...`
    );

    try {
      const { id } = await api.createBenchmark(
        uploadResult.upload_id,
        selectedHarness,
        selectedModel
      );
      router.push(`/benchmarks/${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to create benchmark");
      setUploading(false);
      setUploadProgress("");
    }
  }, [uploadResult, selectedHarness, router]);

  const handleCancel = () => {
    setUploadResult(null);
    setSelectedHarness("harbor");
    setSelectedModel(DEFAULT_MODEL);
    setError(null);
  };

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
    fileInputRef.current?.click();
  };

  // If file is uploaded, show harness selection
  if (uploadResult) {
    return (
      <div className="border-2 border-gray-200 rounded-xl p-8">
        <div className="text-center mb-6">
          <div className="text-5xl mb-4">📦</div>
          <p className="text-gray-700 font-medium text-lg">
            Task: {uploadResult.task_name}
          </p>
        </div>

        <HarnessSelector
          value={selectedHarness}
          onChange={setSelectedHarness}
          detectedFormat={uploadResult.detected_format}
          disabled={uploading}
        />

        <div className="mt-6">
          <ModelSelector
            value={selectedModel}
            onChange={setSelectedModel}
            disabled={uploading}
          />
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600 text-sm">{error}</p>
          </div>
        )}

        <div className="mt-6 flex gap-4">
          <button
            onClick={handleCancel}
            disabled={uploading}
            className="flex-1 py-3 px-4 border border-gray-300 rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={uploading}
            className="flex-1 py-3 px-4 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            {uploading ? uploadProgress : "Start 10 Runs"}
          </button>
        </div>
      </div>
    );
  }

  // Original drop zone UI for file selection
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
