"use client";

import { useCallback, useState, useRef } from "react";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";

// Hardcoded values - terminus and model selection disabled for now
const HARNESS = "harbor";
const MODEL = "openrouter/openai/gpt-5";

export function DropZone() {
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<string>("");

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

      // Immediately start benchmark after upload
      setUploadProgress(`Starting benchmark for "${result.task_name}"...`);

      const { id } = await api.createBenchmark(
        result.upload_id,
        HARNESS,
        MODEL
      );
      router.push(`/benchmarks?id=${id}`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setUploading(false);
      setUploadProgress("");
    }
  }, [router]);

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
