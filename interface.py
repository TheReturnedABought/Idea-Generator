import tkinter as tk
from tkinter import font as tkfont
import random
from generator import generate_coding_idea, get_description

# --- Color Palette ---
BG_DARK     = "#0d0f14"
BG_CARD     = "#161a24"
BG_CARD2    = "#1c2030"
ACCENT      = "#7fffb2"        # neon mint
ACCENT2     = "#4fc3f7"        # electric blue
MUTED       = "#5a6378"
TEXT_MAIN   = "#e8ecf4"
TEXT_SUB    = "#9aa3b5"
BORDER      = "#252a3a"
BTN_BG      = "#1f2d40"
BTN_HOVER   = "#2a3f58"
BTN_PRESS   = "#162030"
TAG_BG      = "#1a2a1e"
TAG_FG      = "#7fffb2"


class IdeaGeneratorGUI:
    def __init__(self, root):
        self.root = root
        root.title("💡 Coding Idea Generator")
        root.configure(bg=BG_DARK)
        root.geometry("860x620")
        root.resizable(False, False)

        # Fonts
        self.font_title   = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.font_heading  = tkfont.Font(family="Courier New", size=9, weight="bold")
        self.font_idea     = tkfont.Font(family="Courier New", size=15, weight="bold")
        self.font_body     = tkfont.Font(family="Georgia", size=11)
        self.font_label    = tkfont.Font(family="Courier New", size=8, weight="bold")
        self.font_btn      = tkfont.Font(family="Courier New", size=11, weight="bold")
        self.font_counter  = tkfont.Font(family="Courier New", size=8)

        self._build_ui()
        self.generate_idea()

    # ──────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self.root, bg=BG_DARK, pady=0)
        header.pack(fill="x", padx=0, pady=0)

        title_frame = tk.Frame(header, bg=BG_DARK)
        title_frame.pack(pady=(20, 6))

        tk.Label(title_frame, text="◈ ", font=self.font_title,
                 bg=BG_DARK, fg=ACCENT).pack(side="left")
        tk.Label(title_frame, text="CODING IDEA GENERATOR",
                 font=self.font_title, bg=BG_DARK, fg=TEXT_MAIN).pack(side="left")
        tk.Label(title_frame, text=" ◈", font=self.font_title,
                 bg=BG_DARK, fg=ACCENT).pack(side="left")

        # Thin accent line
        tk.Frame(self.root, bg=ACCENT, height=1).pack(fill="x", padx=30)

        # ── Idea card ──
        idea_card = tk.Frame(self.root, bg=BG_CARD, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        idea_card.pack(fill="x", padx=30, pady=(18, 0))

        idea_inner = tk.Frame(idea_card, bg=BG_CARD, padx=22, pady=18)
        idea_inner.pack(fill="x")

        # Tag pill
        tag_row = tk.Frame(idea_inner, bg=BG_CARD)
        tag_row.pack(anchor="w", pady=(0, 10))
        tk.Label(tag_row, text=" IDEA ", font=self.font_label,
                 bg=TAG_BG, fg=TAG_FG, padx=6, pady=2).pack(side="left")

        # The idea text — wraplength updated dynamically on resize
        self.idea_var = tk.StringVar()
        self.idea_lbl = tk.Label(idea_inner, textvariable=self.idea_var,
                                  font=self.font_idea, bg=BG_CARD, fg=TEXT_MAIN,
                                  wraplength=760, justify="left", anchor="w")
        self.idea_lbl.pack(fill="x")
        idea_inner.bind("<Configure>", self._update_wraplength)
        # Keep wraplength in sync with actual card width so text never clips
        self.idea_lbl.bind("<Configure>", lambda e: self.idea_lbl.config(wraplength=e.width - 4))

        # Coloured component pills row
        self.pills_frame = tk.Frame(idea_inner, bg=BG_CARD)
        self.pills_frame.pack(anchor="w", pady=(10, 0))

        # ── Bottom bar — packed BEFORE desc_card so expand=True doesn't swallow it ──
        bottom = tk.Frame(self.root, bg=BG_DARK)
        bottom.pack(fill="x", padx=30, pady=(10, 20), side="bottom")

        # ── Description card ──
        desc_card = tk.Frame(self.root, bg=BG_CARD2, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        desc_card.pack(fill="both", expand=True, padx=30, pady=(10, 0))

        desc_inner = tk.Frame(desc_card, bg=BG_CARD2, padx=22, pady=18)
        desc_inner.pack(fill="both", expand=True)

        tag_row2 = tk.Frame(desc_inner, bg=BG_CARD2)
        tag_row2.pack(anchor="w", pady=(0, 10))
        tk.Label(tag_row2, text=" DESCRIPTION ", font=self.font_label,
                 bg="#1a1f30", fg=ACCENT2, padx=6, pady=2).pack(side="left")

        # Arrow decorator
        tk.Label(tag_row2, text="  →", font=self.font_label,
                 bg=BG_CARD2, fg=MUTED).pack(side="left", padx=(8, 0))

        self.desc_text = tk.Text(desc_inner, font=self.font_body,
                                  bg=BG_CARD2, fg=TEXT_SUB,
                                  relief="flat", bd=0, wrap="word",
                                  selectbackground=ACCENT, selectforeground=BG_DARK,
                                  insertbackground=ACCENT,
                                  state="disabled", cursor="arrow",
                                  padx=0, pady=0, spacing2=4)
        self.desc_text.pack(fill="both", expand=True)

        self.btn = tk.Button(
            bottom,
            text="⟳  GENERATE NEW IDEA",
            font=self.font_btn,
            bg=BTN_BG, fg=ACCENT,
            activebackground=BTN_HOVER, activeforeground=ACCENT,
            relief="flat", bd=0, padx=20, pady=10,
            cursor="hand2",
            command=self.generate_idea
        )
        self.btn.pack(side="left", padx=(0, 12))
        self.btn.bind("<Enter>",  lambda e: self.btn.config(bg=BTN_HOVER))
        self.btn.bind("<Leave>",  lambda e: self.btn.config(bg=BTN_BG))
        self.btn.bind("<ButtonPress-1>",   lambda e: self.btn.config(bg=BTN_PRESS))
        self.btn.bind("<ButtonRelease-1>", lambda e: self.btn.config(bg=BTN_HOVER))

        self.counter_lbl = tk.Label(bottom, text="#0001",
                                     font=self.font_counter,
                                     bg=BG_DARK, fg=MUTED)
        self.counter_lbl.pack(side="left", padx=(12, 0))

        self.copy_btn = tk.Button(
            bottom,
            text="⎘  COPY",
            font=self.font_btn,
            bg=BTN_BG, fg=ACCENT2,
            activebackground=BTN_HOVER, activeforeground=ACCENT2,
            relief="flat", bd=0, padx=20, pady=10,
            cursor="hand2",
            command=self._copy_to_clipboard
        )
        self.copy_btn.pack(side="right")
        self.copy_btn.bind("<Enter>",  lambda e: self.copy_btn.config(bg=BTN_HOVER))
        self.copy_btn.bind("<Leave>",  lambda e: self.copy_btn.config(bg=BTN_BG))
        self.copy_btn.bind("<ButtonPress-1>",   lambda e: self.copy_btn.config(bg=BTN_PRESS))
        self.copy_btn.bind("<ButtonRelease-1>", lambda e: self.copy_btn.config(bg=BTN_HOVER))

        self._count = 0

    # ──────────────────────────────────────────────
    def _update_wraplength(self, event):
        """Keep the idea label wraplength flush with the actual card width."""
        # event.width is the inner frame width; subtract horizontal padding (22*2)
        new_wrap = max(200, event.width - 44)
        self.idea_lbl.config(wraplength=new_wrap)

    # ──────────────────────────────────────────────
    def _make_pill(self, parent, text, color, bg_color):
        """Create a small coloured label pill."""
        tk.Label(parent, text=f" {text} ", font=self.font_label,
                 bg=bg_color, fg=color, padx=5, pady=2).pack(side="left", padx=(0, 6))

    # ──────────────────────────────────────────────
    def generate_idea(self):
        self._count += 1
        adj, noun, verb, twist = generate_coding_idea()
        full_idea = f"{adj} {noun} that {verb} {twist}"
        description = get_description(adj, noun, verb, twist)

        # Update idea label
        self.idea_var.set(full_idea)

        # Rebuild pills
        for w in self.pills_frame.winfo_children():
            w.destroy()

        pill_data = [
            (adj,   ACCENT,  "#0e2018"),
            (noun,  ACCENT2, "#0d1c26"),
            (verb,  "#f9c74f", "#251f0a"),
            (twist, "#f77f7f", "#250e0e"),
        ]
        for text, fg, bg in pill_data:
            # Trim if too long
            short = text if len(text) <= 30 else text[:28] + "…"
            self._make_pill(self.pills_frame, short, fg, bg)

        # Update description
        self.desc_text.configure(state="normal")
        self.desc_text.delete("1.0", tk.END)
        self.desc_text.insert(tk.END, description)
        self.desc_text.configure(state="disabled")

        # Update counter
        self.counter_lbl.config(text=f"#{self._count:04d}")

        # Flash accent border on idea card briefly
        self._flash()

    def _copy_to_clipboard(self):
        idea  = self.idea_var.get()
        desc  = self.desc_text.get("1.0", tk.END).strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(f"{idea}\n\n{desc}")
        # Brief visual confirmation
        self.copy_btn.config(text="✓  COPIED", fg=ACCENT)
        self.root.after(1200, lambda: self.copy_btn.config(text="⎘  COPY", fg=ACCENT2))

    def _flash(self):
        card = self.idea_lbl.master.master  # BG_CARD frame
        card.configure(highlightbackground=ACCENT)
        self.root.after(250, lambda: card.configure(highlightbackground=BORDER))


if __name__ == "__main__":
    root = tk.Tk()
    app = IdeaGeneratorGUI(root)
    root.mainloop()