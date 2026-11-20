var E=Object.defineProperty;var k=(p,o,l)=>o in p?E(p,o,{enumerable:!0,configurable:!0,writable:!0,value:l}):p[o]=l;var n=(p,o,l)=>k(p,typeof o!="symbol"?o+"":o,l);this.FlareChatWidget=function(){"use strict";const p=`
:host {
  --flare-primary: #ff5722;
  --flare-primary-hover: #f4511e;
  --flare-primary-glow: rgba(255, 87, 34, 0.4);
  --flare-bg-main: #0d1117;
  --flare-bg-panel: #161b22;
  --flare-bg-card: #1c2128;
  --flare-border: rgba(240, 246, 252, 0.12);
  --flare-text: #f0f6fc;
  --flare-text-muted: #8b949e;
  --flare-accent: #388bfd;
  --flare-font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --flare-radius: 18px;
  --flare-shadow: 0 16px 48px rgba(0, 0, 0, 0.65), 0 0 0 1px var(--flare-border);
  font-family: var(--flare-font);
  color: var(--flare-text);
  box-sizing: border-box;
}

*, *::before, *::after {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

/* Tooltip Badge "🔥 Hỏi Flare AI" */
.flare-fab-badge {
  position: fixed;
  bottom: 30px;
  right: 92px;
  background: var(--flare-bg-panel);
  color: var(--flare-text);
  border: 1px solid rgba(255, 87, 34, 0.5);
  padding: 9px 16px;
  border-radius: 30px;
  font-size: 13.5px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5), 0 0 14px rgba(255, 87, 34, 0.3);
  cursor: pointer;
  z-index: 2147483647;
  white-space: nowrap;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
  animation: badge-float 3s ease-in-out infinite;
}

.flare-fab-badge:hover {
  transform: translateY(-2px) scale(1.04);
  border-color: var(--flare-primary);
  box-shadow: 0 8px 26px rgba(255, 87, 34, 0.5);
}

.flare-fab-badge.hidden {
  opacity: 0;
  transform: translateX(14px) scale(0.85);
  pointer-events: none;
}

.flare-badge-icon {
  font-size: 16px;
  display: inline-flex;
  align-items: center;
  animation: flame-flicker 1.8s infinite ease-in-out;
}

@keyframes badge-float {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

@keyframes flame-flicker {
  0%, 100% { transform: scale(1) rotate(0deg); }
  50% { transform: scale(1.15) rotate(4deg); }
}

/* Nút bấm tròn hình Ngọn lửa (FAB) */
.flare-fab {
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 58px;
  height: 58px;
  border-radius: 50%;
  background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%);
  color: #ffffff;
  border: none;
  cursor: pointer;
  box-shadow: 0 8px 28px rgba(255, 87, 34, 0.55), 0 0 0 2px rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2147483647;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.flare-fab:hover {
  transform: translateY(-3px) scale(1.06);
  box-shadow: 0 12px 34px rgba(255, 87, 34, 0.7), 0 0 20px rgba(255, 87, 34, 0.75);
}

.flare-fab:active {
  transform: translateY(0) scale(0.96);
}

.flare-fab svg {
  width: 30px;
  height: 30px;
  fill: currentColor;
  transition: all 0.3s ease;
}

.flare-fab.open svg.icon-flame {
  display: none;
}

.flare-fab:not(.open) svg.icon-close {
  display: none;
}

.flare-fab-pulse {
  position: absolute;
  width: 100%;
  height: 100%;
  border-radius: 50%;
  border: 2px solid rgba(255, 87, 34, 0.85);
  animation: pulse-ring 2.4s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
  pointer-events: none;
}

@keyframes pulse-ring {
  0% { transform: scale(0.95); opacity: 0.85; }
  50% { transform: scale(1.4); opacity: 0; }
  100% { transform: scale(1.4); opacity: 0; }
}

/* Khung cửa sổ Chat — Mở rộng rộng rãi hơn (440px) */
.flare-window {
  position: fixed;
  bottom: 94px;
  right: 24px;
  width: 440px;
  max-width: calc(100vw - 32px);
  height: 620px;
  max-height: calc(100vh - 115px);
  background: var(--flare-bg-main);
  border: 1px solid var(--flare-border);
  border-radius: var(--flare-radius);
  box-shadow: var(--flare-shadow);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  z-index: 2147483646;
  opacity: 0;
  transform: translateY(16px) scale(0.96);
  pointer-events: none;
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}

.flare-window.active {
  opacity: 1;
  transform: translateY(0) scale(1);
  pointer-events: all;
}

/* Chế độ Toàn màn hình (Fullscreen) */
.flare-window.fullscreen {
  top: 16px;
  bottom: 16px;
  left: 16px;
  right: 16px;
  width: auto;
  max-width: none;
  height: auto;
  max-height: none;
  border-radius: 20px;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.85), 0 0 0 1px rgba(255, 87, 34, 0.3);
}

/* Header */
.flare-header {
  padding: 14px 18px;
  background: var(--flare-bg-panel);
  border-bottom: 1px solid var(--flare-border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-shrink: 0;
}

.flare-header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.flare-header-avatar {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  background: linear-gradient(135deg, rgba(255, 87, 34, 0.25), rgba(216, 67, 21, 0.45));
  border: 1px solid rgba(255, 87, 34, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ff5722;
  box-shadow: 0 0 12px rgba(255, 87, 34, 0.35);
}

.flare-header-avatar svg {
  width: 22px;
  height: 22px;
  fill: currentColor;
}

.flare-header-title {
  font-size: 15.5px;
  font-weight: 600;
  color: var(--flare-text);
  line-height: 1.2;
}

.flare-header-subtitle {
  font-size: 11.5px;
  color: var(--flare-text-muted);
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 3px;
}

.flare-status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #3fb950;
  box-shadow: 0 0 8px rgba(63, 185, 80, 0.9);
}

.flare-header-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.flare-btn-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: transparent;
  border: none;
  color: var(--flare-text-muted);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
}

.flare-btn-icon:hover {
  background: var(--flare-bg-card);
  color: var(--flare-text);
}

.flare-btn-icon svg {
  width: 17px;
  height: 17px;
  fill: currentColor;
}

/* Biểu tượng phóng to/thu nhỏ toàn màn hình */
.flare-window:not(.fullscreen) .icon-compress {
  display: none;
}

.flare-window.fullscreen .icon-expand {
  display: none;
}

/* Khu vực tin nhắn */
.flare-messages {
  flex: 1 1 auto;
  min-height: 0;
  padding: 18px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
  scrollbar-width: thin;
  scrollbar-color: var(--flare-bg-card) transparent;
}

.flare-messages::-webkit-scrollbar {
  width: 6px;
}
.flare-messages::-webkit-scrollbar-thumb {
  background: var(--flare-bg-card);
  border-radius: 3px;
}

.flare-msg {
  display: flex;
  gap: 10px;
  max-width: 90%;
  animation: msg-appear 0.25s ease-out;
}

@keyframes msg-appear {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

.flare-msg.user {
  margin-left: auto;
  flex-direction: row-reverse;
}

.flare-msg.assistant {
  margin-right: auto;
}

.flare-msg-avatar {
  width: 32px;
  height: 32px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
  font-size: 15px;
}

.flare-msg.assistant .flare-msg-avatar {
  background: linear-gradient(135deg, rgba(255, 87, 34, 0.2), rgba(216, 67, 21, 0.4));
  border: 1px solid rgba(255, 87, 34, 0.45);
  color: var(--flare-primary);
}

.flare-msg.assistant .flare-msg-avatar svg {
  width: 18px;
  height: 18px;
  fill: currentColor;
}

.flare-msg.user .flare-msg-avatar {
  background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%);
  color: white;
}

.flare-msg-bubble {
  padding: 12px 16px;
  border-radius: 14px;
  font-size: 13.8px;
  line-height: 1.6;
  word-break: break-word;
}

.flare-msg.user .flare-msg-bubble {
  background: linear-gradient(135deg, #ff5722 0%, #e64a19 100%);
  color: #ffffff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 4px 16px rgba(255, 87, 34, 0.3);
}

.flare-msg.assistant .flare-msg-bubble {
  background: var(--flare-bg-card);
  border: 1px solid var(--flare-border);
  color: var(--flare-text);
  border-bottom-left-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.flare-msg-bubble p:not(:last-child) {
  margin-bottom: 8px;
}

.flare-msg-bubble ul, .flare-msg-bubble ol {
  margin-left: 20px;
  margin-bottom: 8px;
}

.flare-msg-bubble li {
  margin-bottom: 5px;
}

.flare-msg-bubble strong {
  color: #ff9e80;
  font-weight: 600;
}

.flare-msg.user .flare-msg-bubble strong {
  color: #ffffff;
}

.flare-msg-bubble code {
  background: rgba(0, 0, 0, 0.4);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 12.5px;
  color: #ffab91;
}

/* Các Tag Câu Hỏi Nhanh Xếp Hàng Dọc Bên Phải (Như Mẫu Easy AI) */
.flare-suggestions-group {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 9px;
  margin-top: 4px;
  margin-bottom: 8px;
  animation: msg-appear 0.3s ease-out;
}

.flare-suggest-pill {
  padding: 9px 18px;
  font-size: 13px;
  font-weight: 500;
  color: #ff9e80;
  background: rgba(255, 87, 34, 0.08);
  border: 1.5px solid rgba(255, 87, 34, 0.45);
  border-radius: 22px;
  cursor: pointer;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  text-align: right;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

.flare-suggest-pill:hover {
  background: rgba(255, 87, 34, 0.2);
  border-color: #ff5722;
  color: #ffffff;
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 4px 14px rgba(255, 87, 34, 0.35);
}

.flare-suggest-pill:active {
  transform: translateY(0) scale(0.98);
}

.flare-typing {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 12px 18px;
  background: var(--flare-bg-card);
  border: 1px solid var(--flare-border);
  border-radius: 14px;
  width: fit-content;
}

.flare-typing-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--flare-primary);
  animation: typing-jump 1.2s infinite ease-in-out;
}

.flare-typing-dot:nth-child(2) { animation-delay: 0.2s; }
.flare-typing-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes typing-jump {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
  40% { transform: translateY(-6px); opacity: 1; }
}

/* Footer & Input */
.flare-footer {
  padding: 14px 18px;
  background: var(--flare-bg-panel);
  border-top: 1px solid var(--flare-border);
  flex-shrink: 0;
}

.flare-input-wrapper {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--flare-bg-main);
  border: 1px solid var(--flare-border);
  border-radius: 14px;
  padding: 9px 14px;
  transition: all 0.2s ease;
}

.flare-input-wrapper:focus-within {
  border-color: var(--flare-primary);
  box-shadow: 0 0 0 2px rgba(255, 87, 34, 0.25);
}

.flare-input {
  flex: 1;
  background: transparent;
  border: none;
  outline: none;
  color: var(--flare-text);
  font-family: inherit;
  font-size: 13.8px;
  resize: none;
  max-height: 90px;
  line-height: 1.45;
}

.flare-input::placeholder {
  color: #6e7681;
}

.flare-btn-send {
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--flare-primary);
  border: none;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.flare-btn-send:hover:not(:disabled) {
  background: var(--flare-primary-hover);
  transform: scale(1.05);
}

.flare-btn-send:disabled {
  opacity: 0.35;
  cursor: not-allowed;
}

.flare-btn-send svg {
  width: 17px;
  height: 17px;
  fill: currentColor;
}

.flare-brand {
  font-size: 11px;
  color: #6e7681;
  text-align: center;
  margin-top: 8px;
}

@media (max-width: 480px) {
  .flare-window {
    right: 8px;
    left: 8px;
    bottom: 84px;
    width: auto;
    height: calc(100vh - 96px);
  }
  .flare-fab-badge {
    display: none;
  }
}
`;async function o(h,e,t=[],a="guest"){const i=`${h.replace(/\/+$/,"")}/chat`,r=await fetch(i,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:e,history:t,role:a})});if(!r.ok){let s=`Lỗi máy chủ (${r.status})`;try{const c=await r.json();c.detail&&(s=c.detail)}catch{}throw new Error(s)}return r.json()}const l=`
<svg class="icon-flame" viewBox="0 0 24 24" fill="currentColor">
  <path d="M12 23c-4.97 0-9-3.96-9-8.83 0-3.9 2.53-7.58 4.67-9.84.41-.44 1.12-.34 1.39.2 1.09 2.18 2.52 3.82 3.94 4.54.42.21.92-.08.95-.55.15-2.26 1.4-5.4 3.95-7.46.46-.37 1.15-.05 1.17.54.12 3.52 1.39 6.27 3.01 7.82 1.25 1.19 1.92 2.62 1.92 4.58 0 4.87-4.03 8.83-9 8.83z"/>
</svg>
`,d={guest:{subtitle:"Khách vãng lai",greeting:"Bạn cần mình hỗ trợ gì về kỹ năng an toàn PCCC & thoát hiểm ạ? 😊",quickPrompts:["🚨 Kỹ năng thoát hiểm khi có khói độc","🧯 Cách sử dụng bình chữa cháy CO2 và bột","📞 Khi phát hiện cháy cần gọi số nào?","ℹ️ Giới thiệu hệ thống cảnh báo cháy FLARE"]},user:{subtitle:"Người dùng hệ thống",greeting:"Chào bạn! Bạn cần hỗ trợ gì về sử dụng FLARE hoặc an toàn PCCC? 😊",quickPrompts:["📱 Hướng dẫn kết nối Bot Zalo nhận cảnh báo","📹 Cách bật webcam laptop phát hiện cháy","❓ Cách gửi phản hồi khi hệ thống nhận diện sai","🚨 Kỹ năng thoát hiểm khi có khói"]},admin_ops:{subtitle:"Quản trị Vận hành",greeting:"Chào Quản trị viên Vận hành! Bạn cần hỗ trợ gì về tác vụ hệ thống? ⚡",quickPrompts:["📊 Cách giám sát phiên cảnh báo realtime hệ thống","👥 Quy trình quản lý và duyệt tài khoản người dùng","📋 Cách xem báo cáo đánh giá sự cố hỏa hoạn","⚙️ Hướng dẫn lệnh CLI tạo tài khoản quản trị"]},admin:{subtitle:"Quản trị Vận hành",greeting:"Chào Quản trị viên Vận hành! Bạn cần hỗ trợ gì về tác vụ hệ thống? ⚡",quickPrompts:["📊 Cách giám sát phiên cảnh báo realtime hệ thống","👥 Quy trình quản lý và duyệt tài khoản người dùng","📋 Cách xem báo cáo đánh giá sự cố hỏa hoạn","⚙️ Hướng dẫn lệnh CLI tạo tài khoản quản trị"]},admin_model:{subtitle:"Quản trị Mô hình AI",greeting:"Chào Quản trị viên Mô hình! Bạn cần hỗ trợ gì về MLOps & Dataset? 🎯",quickPrompts:["🏷️ Hướng dẫn dùng canvas SVG vẽ lại bounding box","🔄 Chu trình Active Learning bổ sung dữ liệu","📦 Cách xuất dataset chuẩn định dạng YOLO","🧪 Quy trình kiểm định và đối soát mô hình mới"]}};function b(h){if(!h)return"";let t=h.replace(/\[nguồn:\s*[^\]]+\]/gi,"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");t=t.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>"),t=t.replace(/`([^`]+)`/g,"<code>$1</code>");const a=t.split(`
`),i=[];let r=!1,s=!1;for(const c of a){const g=c.trim();if(g.startsWith("- ")||g.startsWith("* ")){s&&(i.push("</ol>"),s=!1),r||(i.push("<ul>"),r=!0),i.push(`<li>${g.substring(2)}</li>`);continue}const u=g.match(/^(\d+)\.\s+(.*)/);if(u){r&&(i.push("</ul>"),r=!1),s||(i.push("<ol>"),s=!0),i.push(`<li>${u[2]}</li>`);continue}r&&(i.push("</ul>"),r=!1),s&&(i.push("</ol>"),s=!1),g.length>0&&i.push(`<p>${c}</p>`)}return r&&i.push("</ul>"),s&&i.push("</ol>"),i.join("")}function C(){var e;if((e=window.FlareChatWidgetConfig)!=null&&e.apiUrl)return window.FlareChatWidgetConfig.apiUrl;const h=document.querySelectorAll("script");for(let t=0;t<h.length;t++){const a=h[t].src||"";if(a.includes("widget.js"))try{return`${new URL(a).origin}/api`}catch{}}return"http://localhost:8001/api"}class x{constructor(e){n(this,"config");n(this,"container",null);n(this,"shadow",null);n(this,"isOpen",!1);n(this,"isFullscreen",!1);n(this,"isLoading",!1);n(this,"currentRole","guest");n(this,"currentUserId",null);n(this,"history",[]);n(this,"messagesContainer",null);n(this,"inputElem",null);n(this,"sendBtn",null);n(this,"fabBtn",null);n(this,"fabBadge",null);n(this,"windowElem",null);n(this,"expandBtn",null);n(this,"subtitleElem",null);var i;const t=((i=window.FlareChatWidgetConfig)==null?void 0:i.role)||(e==null?void 0:e.role)||"guest";this.currentRole=this.normalizeRole(t);const u=((i=window.FlareChatWidgetConfig)==null?void 0:i.userId)||(e==null?void 0:e.userId)||null;this.currentUserId=this.currentRole==="guest"?null:(u?String(u):null);const a=d[this.currentRole]||d.guest;this.config={apiUrl:C(),title:"Flare AI",subtitle:a.subtitle,placeholder:"Hỏi Flare AI...",quickPrompts:a.quickPrompts,...window.FlareChatWidgetConfig,...e},this.mount()}normalizeRole(e){const t=(e||"").toLowerCase().trim();return t==="admin"||t==="admin_ops"?"admin_ops":t==="admin_model"?"admin_model":t==="user"?"user":"guest"}getStorageKey(){return!this.currentUserId||this.currentRole==="guest"?null:"flare_chat_history_"+this.currentUserId}saveHistory(){const e=this.getStorageKey();if(!e)return;try{localStorage.setItem(e,JSON.stringify(this.history.slice(-20)))}catch{}}loadHistory(){if(!this.messagesContainer)return;this.messagesContainer.innerHTML="";const e=this.getStorageKey();let t=[];if(e)try{const a=localStorage.getItem(e);if(a){const i=JSON.parse(a);Array.isArray(i)&&(t=i)}}catch{t=[]}this.history=t,this.renderInitialGreeting();if(t.length>0){for(const a of t){a.role==="user"?this.appendUserMessage(a.content):a.role==="assistant"&&this.appendAssistantMessage(a.content)}}this.scrollToBottom()}setRole(e,u=null){const t=this.normalizeRole(e),uid=t==="guest"?null:(u?String(u):this.currentUserId),changed=this.currentRole!==t||this.currentUserId!==uid;this.currentRole=t,this.currentUserId=uid;const a=d[t]||d.guest;this.config.subtitle=a.subtitle,this.config.quickPrompts=a.quickPrompts,this.subtitleElem&&(this.subtitleElem.textContent=a.subtitle),changed&&this.loadHistory()}init(e){e&&(e.role&&this.setRole(e.role,e.userId),this.config={...this.config,...e})}open(){var e,t,a;this.isOpen||(this.isOpen=!0,(e=this.windowElem)==null||e.classList.add("active"),(t=this.fabBtn)==null||t.classList.add("open"),(a=this.fabBadge)==null||a.classList.add("hidden"),setTimeout(()=>{var i;(i=this.inputElem)==null||i.focus(),this.scrollToBottom()},250))}close(){var e,t,a;this.isOpen&&(this.isOpen=!1,(e=this.windowElem)==null||e.classList.remove("active"),(t=this.fabBtn)==null||t.classList.remove("open"),(a=this.fabBadge)==null||a.classList.remove("hidden"))}toggle(){this.isOpen?this.close():this.open()}toggleFullscreen(){var e,t;this.isFullscreen=!this.isFullscreen,this.isFullscreen?(e=this.windowElem)==null||e.classList.add("fullscreen"):(t=this.windowElem)==null||t.classList.remove("fullscreen"),this.scrollToBottom()}clear(){this.history=[];const e=this.getStorageKey();if(e)try{localStorage.removeItem(e)}catch{}this.messagesContainer&&(this.messagesContainer.innerHTML="",this.renderInitialGreeting())}mount(){if(document.getElementById("flare-rag-widget-root"))return;this.container=document.createElement("div"),this.container.id="flare-rag-widget-root",document.body.appendChild(this.container),this.shadow=this.container.attachShadow({mode:"open"});const e=document.createElement("style");e.textContent=p,this.shadow.appendChild(e);const t=d[this.currentRole]||d.guest,a=document.createElement("div");a.innerHTML=`
      <!-- Tooltip Badge Gọi Mời Nhận Diện Chatbot -->
      <div class="flare-fab-badge" id="flare-fab-badge" title="Bấm để trò chuyện cùng Flare AI">
        <span class="flare-badge-icon">🔥</span>
        <span>Hỏi Flare AI</span>
      </div>

      <!-- Nút Tròn Nổi FAB với Icon Ngọn Lửa -->
      <button class="flare-fab" id="flare-fab" title="Trợ lý Flare AI">
        <div class="flare-fab-pulse"></div>
        ${l}
        <svg class="icon-close" viewBox="0 0 24 24">
          <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
        </svg>
      </button>

      <!-- Khung Cửa Sổ Chatbot -->
      <div class="flare-window" id="flare-window">
        <!-- Header -->
        <div class="flare-header">
          <div class="flare-header-left">
            <div class="flare-header-avatar">
              ${l}
            </div>
            <div>
              <div class="flare-header-title">${this.config.title}</div>
              <div class="flare-header-subtitle">
                <span class="flare-status-dot"></span>
                <span id="flare-header-subtitle-text">${t.subtitle}</span>
              </div>
            </div>
          </div>
          <div class="flare-header-actions">
            <!-- Nút Làm Mới Đoạn Chat -->
            <button class="flare-btn-icon" id="flare-btn-clear" title="Làm mới cuộc trò chuyện">
              <svg viewBox="0 0 24 24">
                <path d="M17.65 6.35C16.2 4.9 14.21 4 12 4c-4.42 0-7.99 3.58-7.99 8s3.57 8 7.99 8c3.73 0 6.84-2.55 7.73-6h-2.08c-.82 2.33-3.04 4-5.65 4-3.31 0-6-2.69-6-6s2.69-6 6-6c1.66 0 3.14.69 4.22 1.78L13 11h7V4l-2.35 2.35z"/>
              </svg>
            </button>

            <!-- Nút Mở Rộng / Toàn Màn Hình -->
            <button class="flare-btn-icon" id="flare-btn-expand" title="Mở rộng / Thu nhỏ toàn màn hình">
              <svg class="icon-expand" viewBox="0 0 24 24">
                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
              </svg>
              <svg class="icon-compress" viewBox="0 0 24 24">
                <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-14v3h3v2h-5V5h2z"/>
              </svg>
            </button>

            <!-- Nút Đóng / Thu Nhỏ -->
            <button class="flare-btn-icon" id="flare-btn-close" title="Thu nhỏ">
              <svg viewBox="0 0 24 24">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
              </svg>
            </button>
          </div>
        </div>

        <!-- Khu vực hiển thị tin nhắn -->
        <div class="flare-messages" id="flare-messages"></div>

        <!-- Khung nhập câu hỏi -->
        <div class="flare-footer">
          <div class="flare-input-wrapper">
            <textarea
              class="flare-input"
              id="flare-input"
              rows="1"
              placeholder="${this.config.placeholder}"
            ></textarea>
            <button class="flare-btn-send" id="flare-btn-send" disabled title="Gửi tin nhắn">
              <svg viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
              </svg>
            </button>
          </div>
          <div class="flare-brand">Hệ thống FLARE • Trợ lý AI Thông minh</div>
        </div>
      </div>
    `,this.shadow.appendChild(a),this.fabBtn=this.shadow.getElementById("flare-fab"),this.fabBadge=this.shadow.getElementById("flare-fab-badge"),this.windowElem=this.shadow.getElementById("flare-window"),this.messagesContainer=this.shadow.getElementById("flare-messages"),this.inputElem=this.shadow.getElementById("flare-input"),this.sendBtn=this.shadow.getElementById("flare-btn-send"),this.expandBtn=this.shadow.getElementById("flare-btn-expand"),this.subtitleElem=this.shadow.getElementById("flare-header-subtitle-text"),this.bindEvents(),this.loadHistory()}renderInitialGreeting(){if(!this.messagesContainer)return;const e=d[this.currentRole]||d.guest,t=document.createElement("div");t.className="flare-msg assistant",t.innerHTML=`
      <div class="flare-msg-avatar">${l}</div>
      <div class="flare-msg-bubble">
        <p>Xin chào, mình là trợ lý ảo <strong>Flare AI</strong> 👋</p>
      </div>
    `,this.messagesContainer.appendChild(t);const a=document.createElement("div");a.className="flare-msg assistant",a.innerHTML=`
      <div class="flare-msg-avatar" style="visibility: hidden;">${l}</div>
      <div class="flare-msg-bubble">
        <p>${e.greeting}</p>
      </div>
    `,this.messagesContainer.appendChild(a);const i=document.createElement("div");i.className="flare-suggestions-group",i.innerHTML=e.quickPrompts.map(r=>`<button class="flare-suggest-pill" data-prompt="${r}">${r}</button>`).join(""),this.messagesContainer.appendChild(i),i.querySelectorAll(".flare-suggest-pill").forEach(r=>{r.addEventListener("click",()=>{const s=r.getAttribute("data-prompt");s&&this.handleSend(s)})}),this.scrollToBottom()}bindEvents(){var e,t,a,i,r,s,c,g,u,w;(e=this.fabBtn)==null||e.addEventListener("click",()=>this.toggle()),(t=this.fabBadge)==null||t.addEventListener("click",()=>this.open()),(i=(a=this.shadow)==null?void 0:a.getElementById("flare-btn-close"))==null||i.addEventListener("click",()=>this.close()),(s=(r=this.shadow)==null?void 0:r.getElementById("flare-btn-clear"))==null||s.addEventListener("click",()=>this.clear()),(c=this.expandBtn)==null||c.addEventListener("click",()=>this.toggleFullscreen()),(g=this.inputElem)==null||g.addEventListener("input",()=>{!this.inputElem||!this.sendBtn||(this.inputElem.style.height="auto",this.inputElem.style.height=Math.min(this.inputElem.scrollHeight,100)+"px",this.sendBtn.disabled=!this.inputElem.value.trim()||this.isLoading)}),(u=this.inputElem)==null||u.addEventListener("keydown",f=>{var m;if(f.key==="Enter"&&!f.shiftKey){f.preventDefault();const y=(m=this.inputElem)==null?void 0:m.value.trim();y&&!this.isLoading&&this.handleSend(y)}}),(w=this.sendBtn)==null||w.addEventListener("click",()=>{var m;const f=(m=this.inputElem)==null?void 0:m.value.trim();f&&!this.isLoading&&this.handleSend(f)})}async handleSend(e){if(!e||this.isLoading)return;this.inputElem&&(this.inputElem.value="",this.inputElem.style.height="auto"),this.sendBtn&&(this.sendBtn.disabled=!0),this.appendUserMessage(e),this.isLoading=!0;const t=this.showTypingIndicator();try{const a=await o(this.config.apiUrl||"http://localhost:8001/api",e,this.history,this.currentRole);t.remove(),this.appendAssistantMessage(a.answer),this.history.push({role:"user",content:e}),this.history.push({role:"assistant",content:a.answer}),this.history=this.history.slice(-20),this.saveHistory()}catch(a){t.remove();const i=a instanceof Error?a.message:"Đã có lỗi xảy ra.";this.appendAssistantMessage(`⚠️ **Không thể kết nối tới máy chủ Flare AI**

Chi tiết: ${i}

*Gợi ý: Hãy đảm bảo Backend RAG đang hoạt động tại ${this.config.apiUrl}*`)}finally{this.isLoading=!1,this.inputElem&&(this.inputElem.disabled=!1,this.inputElem.focus()),this.sendBtn&&this.inputElem&&(this.sendBtn.disabled=!this.inputElem.value.trim()),this.scrollToBottom()}}appendUserMessage(e){if(!this.messagesContainer)return;const t=document.createElement("div");t.className="flare-msg user",t.innerHTML=`
      <div class="flare-msg-avatar">
        <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
          <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
        </svg>
      </div>
      <div class="flare-msg-bubble">${b(e)}</div>
    `,this.messagesContainer.appendChild(t),this.scrollToBottom()}appendAssistantMessage(e,t=[]){if(!this.messagesContainer)return;const a=document.createElement("div");a.className="flare-msg assistant",a.innerHTML=`
      <div class="flare-msg-avatar">
        ${l}
      </div>
      <div class="flare-msg-bubble">
        ${b(e)}
      </div>
    `,this.messagesContainer.appendChild(a),this.scrollToBottom()}showTypingIndicator(){var t;const e=document.createElement("div");return e.className="flare-typing",e.innerHTML=`
      <div class="flare-typing-dot"></div>
      <div class="flare-typing-dot"></div>
      <div class="flare-typing-dot"></div>
    `,(t=this.messagesContainer)==null||t.appendChild(e),this.scrollToBottom(),e}scrollToBottom(){this.messagesContainer&&requestAnimationFrame(()=>{setTimeout(()=>{this.messagesContainer&&(this.messagesContainer.scrollTop=this.messagesContainer.scrollHeight)},30)})}}let _inst=null;x.setRole=function(r,u=null){if(_inst)_inst.setRole(r,u);else window.FlareChatWidgetConfig=Object.assign(window.FlareChatWidgetConfig||{},{role:r,userId:r==="guest"?null:u})};x.open=function(){_inst&&_inst.open()};x.close=function(){_inst&&_inst.close()};x.toggle=function(){_inst&&_inst.toggle()};x.clear=function(){_inst&&_inst.clear()};x.getInstance=function(){return _inst};function v(){if(!_inst){_inst=new x()}window.flareWidgetInstance=_inst;window.flareWidget=_inst;window.FlareChatWidget=_inst;if(window.FlareChatWidgetConfig){const cfgRole=window.FlareChatWidgetConfig.role,cfgUid=window.FlareChatWidgetConfig.userId;if(cfgRole){_inst.setRole(cfgRole,cfgUid)}}}return document.readyState==="loading"?document.addEventListener("DOMContentLoaded",v):v(),x}();
