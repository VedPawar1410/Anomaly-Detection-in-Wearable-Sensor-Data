import Link from "next/link";

const LINKS = [
  { href: "/", label: "Live stream" },
  { href: "/upload", label: "Upload a recording" },
  { href: "/benchmark", label: "Benchmark" },
];

export default function Nav() {
  return (
    <header className="border-b border-stone-200 bg-white">
      <nav className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold text-stone-900">
          Wearable sensor anomaly detection
        </Link>
        <div className="flex gap-4 text-sm text-stone-600">
          {LINKS.map((l) => (
            <Link key={l.href} href={l.href} className="hover:text-stone-900">
              {l.label}
            </Link>
          ))}
        </div>
      </nav>
    </header>
  );
}
