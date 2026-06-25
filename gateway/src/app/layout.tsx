import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Tostal — Sci-data Platform",
  description: "Cloud-hosted geoscience data platform",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}