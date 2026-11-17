import { useRef, useState } from "react";

export type AnnotationBox = {
  class_id: 0 | 1;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

type Point = { x: number; y: number };
type Drag =
  | { kind: "draw"; start: Point }
  | { kind: "move"; index: number; start: Point; original: AnnotationBox }
  | { kind: "resize"; index: number; corner: string; original: AnnotationBox };

type Props = {
  imageUrl: string;
  boxes: AnnotationBox[];
  setBoxes: React.Dispatch<React.SetStateAction<AnnotationBox[]>>;
  drawClass: 0 | 1;
  onDraw: () => void;
};

const clamp = (value: number) => Math.max(0, Math.min(1, value));

function point(event: React.PointerEvent<SVGElement>): Point {
  const svg = event.currentTarget instanceof SVGSVGElement ? event.currentTarget : event.currentTarget.ownerSVGElement;
  if (!svg) return { x: 0, y: 0 };
  const bounds = svg.getBoundingClientRect();
  return {
    x: clamp((event.clientX - bounds.left) / bounds.width),
    y: clamp((event.clientY - bounds.top) / bounds.height),
  };
}

export default function AnnotationEditor({ imageUrl, boxes, setBoxes, drawClass, onDraw }: Props) {
  const drag = useRef<Drag | null>(null);
  const [draft, setDraft] = useState<{ start: Point; end: Point } | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [mode, setMode] = useState<"draw" | "edit">("draw");

  function beginDraw(event: React.PointerEvent<SVGSVGElement>) {
    if (mode !== "draw") return;
    const start = point(event);
    drag.current = { kind: "draw", start };
    setDraft({ start, end: start });
    setSelected(null);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function beginEdit(event: React.PointerEvent<SVGElement>, next: Exclude<Drag, { kind: "draw" }>) {
    if (mode !== "edit") return;
    event.stopPropagation();
    drag.current = next;
    setSelected(next.index);
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function move(event: React.PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (!active) return;
    const current = point(event);
    if (active.kind === "draw") {
      setDraft({ start: active.start, end: current });
      return;
    }
    if (active.kind === "move") {
      const dx = Math.max(-active.original.x1, Math.min(1 - active.original.x2, current.x - active.start.x));
      const dy = Math.max(-active.original.y1, Math.min(1 - active.original.y2, current.y - active.start.y));
      setBoxes((items) => items.map((box, index) => index === active.index ? {
        ...active.original,
        x1: active.original.x1 + dx,
        x2: active.original.x2 + dx,
        y1: active.original.y1 + dy,
        y2: active.original.y2 + dy,
      } : box));
      return;
    }
    const updated = { ...active.original };
    if (active.corner.includes("left")) updated.x1 = Math.min(current.x, updated.x2 - 0.005);
    if (active.corner.includes("right")) updated.x2 = Math.max(current.x, updated.x1 + 0.005);
    if (active.corner.includes("top")) updated.y1 = Math.min(current.y, updated.y2 - 0.005);
    if (active.corner.includes("bottom")) updated.y2 = Math.max(current.y, updated.y1 + 0.005);
    setBoxes((items) => items.map((box, index) => index === active.index ? updated : box));
  }

  function finish(event: React.PointerEvent<SVGSVGElement>) {
    const active = drag.current;
    if (!active) return;
    if (active.kind === "draw") {
      const end = point(event);
      const box: AnnotationBox = {
        class_id: drawClass,
        x1: Math.min(active.start.x, end.x),
        y1: Math.min(active.start.y, end.y),
        x2: Math.max(active.start.x, end.x),
        y2: Math.max(active.start.y, end.y),
      };
      if (box.x2 - box.x1 >= 0.01 && box.y2 - box.y1 >= 0.01) {
        setBoxes((items) => [...items, box]);
        onDraw();
      }
    }
    drag.current = null;
    setDraft(null);
  }

  const corners = (box: AnnotationBox) => [
    ["left-top", box.x1, box.y1],
    ["right-top", box.x2, box.y1],
    ["left-bottom", box.x1, box.y2],
    ["right-bottom", box.x2, box.y2],
  ] as const;

  return <div className="annotation-editor">
    <div className="annotation-tools" role="group" aria-label="Chế độ gắn nhãn">
      <button type="button" className={mode === "draw" ? "active" : ""} aria-pressed={mode === "draw"} onClick={() => { setMode("draw"); setSelected(null); }}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle", marginRight: "6px" }}><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M12 8v8"/><path d="M8 12h8"/></svg>
        Vẽ box mới
      </button>
      <button type="button" className={mode === "edit" ? "active" : ""} aria-pressed={mode === "edit"} onClick={() => setMode("edit")}>
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ display: "inline-block", verticalAlign: "middle", marginRight: "6px" }}><path d="m5 9-3 3 3 3"/><path d="m9 5 3-3 3 3"/><path d="m19 9 3 3-3 3"/><path d="m9 19 3 3 3-3"/><path d="M2 12h20"/><path d="M12 2v20"/></svg>
        Di chuyển hoặc đổi kích thước
      </button>
    </div>
    <div className="annotation-canvas">
      <img src={imageUrl} alt="Ảnh gốc cần gắn nhãn" draggable={false} />
      <svg className={mode === "draw" ? "drawing" : "editing"} viewBox="0 0 1000 1000" preserveAspectRatio="none" onPointerDown={beginDraw} onPointerMove={move} onPointerUp={finish} onPointerCancel={() => { drag.current = null; setDraft(null); }}>
        <rect x={0} y={0} width={1000} height={1000} fill="transparent" />
        {boxes.map((box, index) => <g key={index}>
          <rect x={box.x1 * 1000} y={box.y1 * 1000} width={(box.x2 - box.x1) * 1000} height={(box.y2 - box.y1) * 1000} className={`annotation-box ${box.class_id === 1 ? "fire" : "smoke"} ${selected === index ? "selected" : ""}`} onPointerDown={mode === "edit" ? (event) => beginEdit(event, { kind: "move", index, start: point(event), original: box }) : undefined} />
          <text x={box.x1 * 1000 + 5} y={Math.max(box.y1 * 1000 - 5, 20)} className="annotation-label">{box.class_id === 1 ? "Lửa" : "Khói"} {index + 1}</text>
          {mode === "edit" && selected === index && corners(box).map(([corner, x, y]) => <circle key={corner} cx={x * 1000} cy={y * 1000} r={9} className="annotation-handle" onPointerDown={(event) => beginEdit(event, { kind: "resize", index, corner, original: box })} />)}
        </g>)}
        {draft && <rect x={Math.min(draft.start.x, draft.end.x) * 1000} y={Math.min(draft.start.y, draft.end.y) * 1000} width={Math.abs(draft.end.x - draft.start.x) * 1000} height={Math.abs(draft.end.y - draft.start.y) * 1000} className="annotation-draft" />}
      </svg>
    </div>
    <p className="small muted">Ở chế độ vẽ, kéo trên ảnh để thêm nhiều box, kể cả trong vùng đã có box. Chuyển sang chỉnh sửa để di chuyển hoặc kéo góc box.</p>
    {boxes.length > 0 && <div className="annotation-list">{boxes.map((box, index) => <div key={index} className={selected === index ? "annotation-row selected" : "annotation-row"}>
      <button type="button" className="text-button" onClick={() => setSelected(index)}>Box {index + 1}</button>
      <select aria-label={`Lớp box ${index + 1}`} value={box.class_id} onChange={(event) => setBoxes((items) => items.map((value, at) => at === index ? { ...value, class_id: Number(event.target.value) as 0 | 1 } : value))}><option value={0}>Khói</option><option value={1}>Lửa</option></select>
      <button type="button" className="text-button danger-text" onClick={() => { setBoxes((items) => items.filter((_, at) => at !== index)); setSelected(null); }}>Xóa box</button>
    </div>)}</div>}
  </div>;
}
