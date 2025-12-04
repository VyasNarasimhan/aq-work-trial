/**
 * Format a date string to the user's local timezone with consistent formatting.
 *
 * @param dateString - ISO date string from the API
 * @param options - Optional Intl.DateTimeFormatOptions to customize output
 * @returns Formatted date string in user's local timezone
 */
export function formatDate(
  dateString: string | null | undefined,
  options?: Intl.DateTimeFormatOptions
): string {
  if (!dateString) {
    return "N/A";
  }

  try {
    // If the date string doesn't have timezone info, treat it as UTC
    // This handles backend dates like "2025-12-04T22:30:00" (no Z suffix)
    let normalizedDateString = dateString;
    if (
      !dateString.endsWith("Z") &&
      !dateString.includes("+") &&
      !dateString.match(/-\d{2}:\d{2}$/)
    ) {
      normalizedDateString = dateString + "Z";
    }

    const date = new Date(normalizedDateString);

    // Check for invalid date
    if (isNaN(date.getTime())) {
      return "Invalid date";
    }

    // Default options for consistent display
    const defaultOptions: Intl.DateTimeFormatOptions = {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
    };

    return date.toLocaleString(undefined, options ?? defaultOptions);
  } catch {
    return "Invalid date";
  }
}

/**
 * Format a date to show relative time (e.g., "2 hours ago")
 */
export function formatRelativeTime(dateString: string | null | undefined): string {
  if (!dateString) {
    return "N/A";
  }

  try {
    // If the date string doesn't have timezone info, treat it as UTC
    let normalizedDateString = dateString;
    if (
      !dateString.endsWith("Z") &&
      !dateString.includes("+") &&
      !dateString.match(/-\d{2}:\d{2}$/)
    ) {
      normalizedDateString = dateString + "Z";
    }

    const date = new Date(normalizedDateString);
    if (isNaN(date.getTime())) {
      return "Invalid date";
    }

    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);

    if (diffSec < 60) {
      return "just now";
    } else if (diffMin < 60) {
      return `${diffMin} minute${diffMin !== 1 ? "s" : ""} ago`;
    } else if (diffHour < 24) {
      return `${diffHour} hour${diffHour !== 1 ? "s" : ""} ago`;
    } else if (diffDay < 7) {
      return `${diffDay} day${diffDay !== 1 ? "s" : ""} ago`;
    } else {
      return formatDate(dateString, {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    }
  } catch {
    return "Invalid date";
  }
}

/**
 * Format a date for compact display (no seconds, includes timezone)
 */
export function formatDateCompact(dateString: string | null | undefined): string {
  return formatDate(dateString, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    timeZoneName: "short",
  });
}
