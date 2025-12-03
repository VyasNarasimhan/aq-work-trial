"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "../api";
import type { Benchmark } from "@/types";

export function useBenchmark(id: string, pollInterval = 3000) {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchBenchmark = useCallback(async () => {
    try {
      const data = await api.getBenchmark(id);
      setBenchmark(data);
      setError(null);
      return data;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      return null;
    } finally {
      setLoading(false);
    }
  }, [id]);

  useEffect(() => {
    let mounted = true;
    let timeoutId: NodeJS.Timeout;

    const poll = async () => {
      const data = await fetchBenchmark();

      if (!mounted) return;

      // Continue polling if still running
      if (data && (data.status === "pending" || data.status === "running")) {
        timeoutId = setTimeout(poll, pollInterval);
      }
    };

    poll();

    return () => {
      mounted = false;
      clearTimeout(timeoutId);
    };
  }, [fetchBenchmark, pollInterval]);

  return { benchmark, error, loading, refetch: fetchBenchmark };
}
