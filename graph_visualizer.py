"""
Knowledge Graph Visualizer for PDF Chat Assistant

Renders an interactive knowledge graph from LightRAG's graphml data
using networkx + pyvis, displayed inside a Streamlit tab.
"""

import os
import tempfile
import streamlit as st
import streamlit.components.v1 as components

GRAPHML_PATH = "./lightrag_storage/graph_chunk_entity_relation.graphml"

ENTITY_COLORS = {
    "person": "#4FC3F7",
    "organization": "#81C784",
    "geo": "#FFB74D",
    "category": "#CE93D8",
    "event": "#F06292",
    "technology": "#4DD0E1",
    "concept": "#AED581",
}

DEFAULT_COLOR = "#90A4AE"


def render_knowledge_graph():
    """Render the knowledge graph visualization inside a Streamlit container."""
    if not os.path.exists(GRAPHML_PATH):
        st.info("Knowledge graph data is not available yet. Upload a PDF using LightRAG mode to generate the graph.")
        return

    file_size = os.path.getsize(GRAPHML_PATH)
    if file_size < 100:
        st.info("Knowledge graph is empty. Process a document first.")
        return

    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError as e:
        st.error(f"Required libraries not available: {e}")
        return

    try:
        G = nx.read_graphml(GRAPHML_PATH)
    except Exception as e:
        st.error(f"Error reading graph data: {e}")
        return

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        st.info("Knowledge graph is empty.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entities", num_nodes)
    with col2:
        st.metric("Relationships", num_edges)
    with col3:
        entity_types = set()
        for _, data in G.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            entity_types.add(etype)
        st.metric("Entity Types", len(entity_types))

    max_display = st.slider(
        "Max nodes to display",
        min_value=10,
        max_value=min(num_nodes, 500),
        value=min(num_nodes, 100),
        step=10,
        help="Limit the number of nodes shown for performance"
    )

    filter_type = st.multiselect(
        "Filter by entity type:",
        options=sorted(entity_types),
        default=sorted(entity_types),
        help="Select which entity types to display"
    )

    if max_display < num_nodes or len(filter_type) < len(entity_types):
        subgraph_nodes = []
        for node, data in G.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            if etype in filter_type:
                subgraph_nodes.append(node)
            if len(subgraph_nodes) >= max_display:
                break
        G_display = G.subgraph(subgraph_nodes).copy()
    else:
        G_display = G

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
        select_menu=False,
        filter_menu=False,
        notebook=False,
    )

    net.barnes_hut(
        gravity=-3000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.01,
        damping=0.09,
        overlap=0,
    )

    for node, data in G_display.nodes(data=True):
        entity_type = data.get("entity_type", "unknown")
        description = data.get("description", "")
        color = ENTITY_COLORS.get(entity_type, DEFAULT_COLOR)

        label = node if len(node) <= 30 else node[:27] + "..."
        title = f"<b>{node}</b><br>Type: {entity_type}"
        if description:
            desc_short = description[:200] + "..." if len(description) > 200 else description
            title += f"<br><br>{desc_short}"

        degree = G_display.degree(node)
        size = max(15, min(50, 15 + degree * 3))

        net.add_node(
            node,
            label=label,
            title=title,
            color=color,
            size=size,
            borderWidth=2,
            borderWidthSelected=4,
        )

    for source, target, data in G_display.edges(data=True):
        description = data.get("description", "")
        keywords = data.get("keywords", "")
        weight = float(data.get("weight", 1.0))

        title = ""
        if keywords:
            title += f"<b>Keywords:</b> {keywords}"
        if description:
            desc_short = description[:150] + "..." if len(description) > 150 else description
            title += f"<br>{desc_short}"

        edge_width = max(1, min(5, weight))

        net.add_edge(
            source,
            target,
            title=title if title else None,
            width=edge_width,
            color="#555555",
        )

    net.set_options("""
    {
        "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "zoomView": true,
            "dragView": true,
            "navigationButtons": true
        }
    }
    """)

    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8")
    try:
        net.save_graph(tmp_file.name)
        with open(tmp_file.name, "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=620, scrolling=False)
    except Exception as e:
        st.error(f"Error rendering graph: {e}")
    finally:
        try:
            os.unlink(tmp_file.name)
        except:
            pass

    legend_items = []
    for etype in sorted(filter_type):
        color = ENTITY_COLORS.get(etype, DEFAULT_COLOR)
        legend_items.append(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background-color:{color};border-radius:50%;margin-right:4px;"></span>'
            f'<span style="margin-right:12px;">{etype}</span>'
        )

    st.markdown(
        '<div style="margin-top:8px;">' + " ".join(legend_items) + '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Entity details", expanded=False):
        entity_list = []
        for node, data in G_display.nodes(data=True):
            entity_list.append({
                "Entity": node,
                "Type": data.get("entity_type", "unknown"),
                "Description": (data.get("description", "")[:100] + "...") if len(data.get("description", "")) > 100 else data.get("description", ""),
                "Connections": G_display.degree(node),
            })
        entity_list.sort(key=lambda x: x["Connections"], reverse=True)

        if entity_list:
            st.dataframe(entity_list, use_container_width=True)
