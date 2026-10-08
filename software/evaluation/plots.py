"""Interactive Plotly figures for the expert KB and RAG retrieval (notebook visualisation cell).

  fig_umap_kb             2-D UMAP of the expert KB, coloured by drug class
  fig_similarity_heatmap  similarity of retrieved reference cases per question
  fig_score_comparison    expert vs DeepSeek no-RAG vs with-RAG scores
  fig_class_distribution  drug-class distribution of the KB

Requires `plotly` and `umap-learn` (see requirements.txt).
"""
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Colour palette (matching the notebook theme) ──────────────────────────────
_PAL = px.colors.qualitative.Pastel + px.colors.qualitative.Safe

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — 2-D UMAP of the Expert KB, coloured by drug class
# ══════════════════════════════════════════════════════════════════════════════
def fig_umap_kb(records: list) -> go.Figure:
    import umap

    df = pd.DataFrame([{
        'drug':       r['drug'],
        'pt':         r['pt'],
        'drug_class': r.get('drug_class', 'UNKNOWN'),
        'case_id':    r['case_id'],
        'no_dates':   r.get('no_dates', False),
        'text':       f"{r['drug']} {r['pt']} {r.get('narrative','')[:120]}",
    } for r in records])

    vec  = TfidfVectorizer(ngram_range=(1,2), max_features=3000)
    mat  = vec.fit_transform(df['text']).toarray()
    emb  = umap.UMAP(n_components=2, random_state=42,
                     n_neighbors=15, min_dist=0.1).fit_transform(mat)

    df['x'] = emb[:, 0]
    df['y'] = emb[:, 1]

    classes      = df['drug_class'].unique().tolist()
    colour_map   = {c: _PAL[i % len(_PAL)] for i, c in enumerate(classes)}
    df['colour'] = df['drug_class'].map(colour_map)

    fig = go.Figure()
    for cls in classes:
        sub = df[df['drug_class'] == cls]
        fig.add_trace(go.Scatter(
            x=sub['x'], y=sub['y'],
            mode='markers',
            name=cls,
            marker=dict(color=colour_map[cls], size=5, opacity=0.75,
                        line=dict(width=0.4, color='rgba(0,0,0,0.25)')),
            text=sub.apply(
                lambda r: f"<b>{r['drug']}</b><br>AE: {r['pt']}<br>"
                          f"Class: {r['drug_class']}<br>Case: {r['case_id']}", axis=1),
            hoverinfo='text',
        ))

    fig.update_layout(
        title='Expert KB — 2-D UMAP (TF-IDF + drug · AE · narrative)',
        xaxis_title='UMAP 1', yaxis_title='UMAP 2',
        legend=dict(title='Drug class', font_size=10,
                    itemsizing='constant', tracegroupgap=2),
        height=600,
        margin=dict(l=40, r=20, t=60, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='white',
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Retrieval similarity heatmap
# Shows: for the current rag_results, how similar were the retrieved
# reference cases to the query pair (per Q and per reference slot)
# ══════════════════════════════════════════════════════════════════════════════
def fig_similarity_heatmap(rag_results: list) -> go.Figure:
    q_names = ['Q2', 'Q5', 'Q10']
    rows, sims, tiers, labels, pairs = [], [], [], [], []

    for entry in rag_results:
        pair_label = f"{entry['drug']}\n{entry['pt'][:28]}"
        for qname in q_names:
            q = entry.get(qname, {})
            refs = q.get('retrieved_refs', [])
            for slot, ref in enumerate(refs[:3], 1):
                rows.append(f"{pair_label} | {qname} | ref {slot}")
                sims.append(round(ref.get('_sim', 0) * 100, 1))
                tier = ref.get('_sim_tier', 'global')
                tier_label = (
                    'same drug'  if tier == 'exact_drug' else
                    'same class' if tier == 'same_class' else
                    'no-date pool' if tier == 'no_date_pool' else 'global'
                )
                tiers.append(tier_label)
                labels.append(
                    f"{ref['drug'][:20]} / {ref['pt'][:20]}<br>"
                    f"Expert {qname}={ref.get(f'{qname}_h_score','?')}<br>"
                    f"sim={ref.get('_sim',0):.0%} [{tier_label}]"
                )

    if not rows:
        fig = go.Figure()
        fig.add_annotation(text='No retrieval data — run the agent first.',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font_size=14)
        fig.update_layout(title='Retrieval similarity', height=250)
        return fig

    tier_colours = {
        'same drug':    '#4C9BE8',
        'same class':   '#F4A261',
        'no-date pool': '#7EC8A4',
        'global':       '#C8A4D4',
    }

    df = pd.DataFrame({'row': rows, 'sim': sims, 'tier': tiers, 'label': labels})
    colours = df['tier'].map(tier_colours).tolist()

    fig = go.Figure(go.Bar(
        x=df['sim'],
        y=df['row'],
        orientation='h',
        marker_color=colours,
        text=df['sim'].apply(lambda v: f'{v:.0f}%'),
        textposition='outside',
        hovertext=df['label'],
        hoverinfo='text',
    ))

    # legend patches
    for tier, colour in tier_colours.items():
        fig.add_trace(go.Bar(
            x=[None], y=[None], orientation='h',
            marker_color=colour, name=tier, showlegend=True,
        ))

    fig.update_layout(
        title='Retrieved reference similarity by query + slot',
        xaxis=dict(title='Cosine similarity (%)', range=[0, 115]),
        height=max(280, len(rows) * 34 + 80),
        barmode='overlay',
        margin=dict(l=260, r=60, t=60, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='white',
        legend=dict(title='Retrieval tier', orientation='v'),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Expert vs AI no-RAG vs AI with-RAG score comparison
# ══════════════════════════════════════════════════════════════════════════════
def _parse_score(text: str):
    if not text:
        return None
    m = re.search(r'Score\s*[:\-]\s*([+-]?\d+(?:\.\d+)?)', str(text), re.IGNORECASE)
    return float(m.group(1)) if m else None

def fig_score_comparison(rag_results: list) -> go.Figure:
    q_names = ['Q2', 'Q5', 'Q10']
    records  = []

    for entry in rag_results:
        pair = f"{entry['drug']} / {entry['pt'][:24]}"
        for qname in q_names:
            q = entry.get(qname, {})
            exact = q.get('expert_exact') or {}

            expert_score  = exact.get('h_score')
            no_rag_score  = _parse_score(q.get('version_no_rag', ''))
            with_rag_score= _parse_score(q.get('version_with_rag', ''))

            records.append({
                'pair':  pair,
                'Q':     qname,
                'Expert (same case)': float(expert_score) if expert_score is not None else None,
                'AI — no RAG':        no_rag_score,
                'AI — with RAG':      with_rag_score,
            })

    if not records:
        fig = go.Figure()
        fig.add_annotation(text='No rag_results data available.',
                           xref='paper', yref='paper', x=0.5, y=0.5,
                           showarrow=False, font_size=14)
        fig.update_layout(title='Score comparison', height=250)
        return fig

    df = pd.DataFrame(records)
    x_labels = df.apply(lambda r: f"{r['pair']}<br>{r['Q']}", axis=1).tolist()

    colours = {
        'Expert (same case)': '#4C9BE8',
        'AI — no RAG':        '#E07B5A',
        'AI — with RAG':      '#5CB87E',
    }

    fig = go.Figure()
    for col, colour in colours.items():
        fig.add_trace(go.Bar(
            name=col,
            x=x_labels,
            y=df[col],
            marker_color=colour,
            text=df[col].apply(lambda v: f'{v:+.0f}' if v is not None else '—'),
            textposition='outside',
            opacity=0.85,
        ))

    fig.update_layout(
        title='Expert vs AI no-RAG vs AI with-RAG — question scores',
        barmode='group',
        yaxis=dict(title='Score', zeroline=True, zerolinewidth=1,
                   zerolinecolor='rgba(0,0,0,0.2)',
                   tickvals=[-2, -1, 0, 1, 2],
                   ticktext=['-2', '-1', '0', '+1', '+2']),
        height=420,
        margin=dict(l=40, r=20, t=60, b=100),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Drug-class distribution of the KB
# ══════════════════════════════════════════════════════════════════════════════
def fig_class_distribution(records: list) -> go.Figure:
    from collections import Counter
    class_counts = Counter(r.get('drug_class', 'UNKNOWN') for r in records)
    df = (pd.DataFrame.from_dict(class_counts, orient='index', columns=['n'])
            .reset_index()
            .rename(columns={'index': 'drug_class'})
            .sort_values('n', ascending=True)
            .tail(30))   # top 30

    colours = [_PAL[i % len(_PAL)] for i in range(len(df))]

    fig = go.Figure(go.Bar(
        x=df['n'], y=df['drug_class'],
        orientation='h',
        marker_color=colours,
        text=df['n'],
        textposition='outside',
    ))
    fig.update_layout(
        title='KB drug-class distribution (top 30)',
        xaxis_title='Number of records',
        height=max(400, len(df) * 22 + 80),
        margin=dict(l=240, r=60, t=60, b=40),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='white',
    )
    return fig


def render_all(kb_records: list, rag_results: list, out_dir=None):
    """Build all four figures and save them as HTML files in output/figures/."""
    from pathlib import Path
    from .. import config
    out_dir = Path(out_dir or config.FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        "kb_umap": fig_umap_kb(kb_records),
        "retrieval_similarity": fig_similarity_heatmap(rag_results),
        "score_comparison": fig_score_comparison(rag_results),
        "kb_class_distribution": fig_class_distribution(kb_records),
    }
    for name, fig in figs.items():
        fig.write_html(str(out_dir / f"{name}.html"))
    return figs
