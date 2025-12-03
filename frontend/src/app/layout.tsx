import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Harbor Benchmark Runner",
  description: "Run AI agent benchmarks on your tasks",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased bg-gray-50 min-h-screen">{children}</body>
    </html>
  );
}
