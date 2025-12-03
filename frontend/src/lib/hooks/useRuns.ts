"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "../api";
import type { Run } from "@/types";

export function useRuns(benchmarkId: string, pollInterval = 0) {
  const [runs, setRuns] = useState<Run[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const fetchRuns = useCallback(async () => {
    try {
      const data = await api.getRuns(benchmarkId);
      setRuns(data);
      setError(null);
      return data;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
      return null;
    } finally {
      setLoading(false);
    }
  }, [benchmarkId]);

  useEffect(() => {
    let mounted = true;
    let timeoutId: NodeJS.Timeout;

    const poll = async () => {
      await fetchRuns();

      if (!mounted) return;

      // Continue polling if interval is set
      if (pollInterval > 0) {
        timeoutId = setTimeout(poll, pollInterval);
      }
    };

    poll();

    return () => {
      mounted = false;
      clearTimeout(timeoutId);
    };
  }, [fetchRuns, pollInterval]);

  return { runs, error, loading, refetch: fetchRuns };
}
