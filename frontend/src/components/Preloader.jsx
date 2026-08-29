import "./Preloader.css";

export default function Preloader() {
  return (
    <div className="preloader">
      <div className="holo-rays" aria-hidden="true" />
      <div className="preloader-core">
        <div className="portal-ring ring-1" aria-hidden="true" />
        <div className="portal-ring ring-2" aria-hidden="true" />
        <div className="portal-ring ring-3" aria-hidden="true" />
        <div className="portal-flash" aria-hidden="true" />
        <div className="portal-crest">
          <span>BAU</span>
        </div>
      </div>
    </div>
  );
}
