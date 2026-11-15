import ApiStatus from "@/components/ApiStatus";
import UploadPanel from "@/components/UploadPanel";

export default function UploadPage() {
  return (
    <div className="space-y-5">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold text-stone-900">Upload a recording</h1>
        <p className="max-w-3xl text-sm text-stone-600">
          Send a PAMAP2-format recording to the API and get back the readings that look faulty, ranked by how far past
          their threshold they are.
        </p>
        <ApiStatus />
      </div>
      <UploadPanel />
    </div>
  );
}
