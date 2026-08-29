import Scene from "./components/Scene";
import TitleCard from "./components/TitleCard";
import ChatPanel from "./components/ChatPanel";
import Preloader from "./components/Preloader";
import "./App.css";

export default function App() {
  return (
    <div className="stage">
      <Scene />
      <div className="grain" />
      <TitleCard />
      <ChatPanel />
      <Preloader />
    </div>
  );
}
