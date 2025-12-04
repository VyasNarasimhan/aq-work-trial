/**
 * Application configuration
 *
 * Backend URL can be overridden via environment variable:
 * NEXT_PUBLIC_API_URL=http://localhost:5001
 */

export const config = {
  // Backend API URL - use env var or default to EC2 instance
  apiUrl: process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001",
  // apiUrl: process.env.NEXT_PUBLIC_API_URL || "http://ec2-54-184-121-95.us-west-2.compute.amazonaws.com:5001",
} as const;

// Convenience export for API base
export const API_BASE = `${config.apiUrl}/api`;
