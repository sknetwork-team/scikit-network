use sprs::CsMat;
use std::fs::File;
use std::fmt::Write as FmtWrite;
use std::io::Write;

use crate::visualization::colors::STANDARD_COLORS;

/// Computes min max scaling.
pub fn min_max_scaling(values: &[f64], x_min: Option<f64>, x_max: Option<f64>) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let min_v = x_min.unwrap_or_else(|| values.iter().copied().fold(f64::INFINITY, f64::min));
    let max_v = x_max.unwrap_or_else(|| values.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    if max_v <= min_v {
        return vec![0.5; values.len()];
    }
    values.iter().map(|v| (v - min_v) / (max_v - min_v)).collect()
}

/// Returns label colors.
pub fn get_label_colors(label_colors: Option<&[String]>) -> Vec<String> {
    match label_colors {
        Some(v) if !v.is_empty() => v.to_vec(),
        _ => STANDARD_COLORS.iter().map(|s| s.to_string()).collect(),
    }
}

/// Returns node colors.
pub fn get_node_colors(
    n: usize,
    labels: Option<&[i32]>,
    scores: Option<&[f64]>,
    node_color: &str,
    label_colors: Option<&[String]>,
) -> Vec<String> {
    let mut node_colors = vec![node_color.to_string(); n];
    if let Some(labs) = labels {
        let colors = get_label_colors(label_colors);
        for i in 0..n.min(labs.len()) {
            let lab = labs[i];
            if lab >= 0 {
                node_colors[i] = colors[(lab as usize) % colors.len()].clone();
            }
        }
        return node_colors;
    }
    if let Some(sc) = scores {
        let palette = crate::visualization::colors::COOLWARM_RGB;
        let palette_svg: Vec<String> = palette
            .iter()
            .map(|c| format!("rgb({}, {}, {})", c[0], c[1], c[2]))
            .collect();
        let scaled = min_max_scaling(sc, None, None);
        for i in 0..n.min(scaled.len()) {
            let idx = (scaled[i] * (palette_svg.len().saturating_sub(1)) as f64) as usize;
            node_colors[i] = palette_svg[idx.min(palette_svg.len().saturating_sub(1))].clone();
        }
    }
    node_colors
}

/// Computes svg text.
pub fn svg_text(
    mut pos: [f64; 2],
    text: &str,
    margin_text: f64,
    font_size: usize,
    position: &str,
) -> String {
    let anchor = match position {
        "left" => {
            pos[0] -= margin_text;
            "end"
        }
        "above" => {
            pos[1] -= margin_text;
            "middle"
        }
        "below" => {
            pos[1] += 2.0 * margin_text;
            "middle"
        }
        _ => {
            pos[0] += margin_text;
            "start"
        }
    };
    let t = escape_xml_text(text);
    format!(
        "<text text-anchor=\"{}\" x=\"{}\" y=\"{}\" font-size=\"{}\">{}</text>",
        anchor, pos[0] as i32, pos[1] as i32, font_size, t
    )
}

fn escape_xml_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn marker_id_from_color(color: &str) -> String {
    let mut slug = String::with_capacity(color.len());
    for ch in color.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
        } else {
            slug.push('_');
        }
    }
    while slug.contains("__") {
        slug = slug.replace("__", "_");
    }
    let slug = slug.trim_matches('_');
    let mut hash: u64 = 5381;
    for b in color.as_bytes() {
        hash = ((hash << 5).wrapping_add(hash)).wrapping_add(*b as u64);
    }
    if slug.is_empty() {
        format!("{:x}", hash)
    } else {
        format!("{}-{:x}", slug, hash)
    }
}

/// Computes svg node.
pub fn svg_node(pos_node: [f64; 2], size: f64, color: &str, stroke_width: f64, stroke_color: &str) -> String {
    format!(
        "<circle cx=\"{}\" cy=\"{}\" r=\"{}\" style=\"fill:{};stroke:{};stroke-width:{}\"/>\n",
        pos_node[0] as i32, pos_node[1] as i32, size, color, stroke_color, stroke_width
    )
}

/// Computes svg edge.
pub fn svg_edge(pos_1: [f64; 2], pos_2: [f64; 2], edge_width: f64, edge_color: &str) -> String {
    format!(
        "<path stroke-width=\"{}\" stroke=\"{}\" d=\"M {} {} {} {}\"/>\n",
        edge_width, edge_color, pos_1[0] as i32, pos_1[1] as i32, pos_2[0] as i32, pos_2[1] as i32
    )
}

/// Computes svg edge directed.
pub fn svg_edge_directed(
    pos_1: [f64; 2],
    pos_2: [f64; 2],
    edge_width: f64,
    edge_color: &str,
    node_size: f64,
) -> String {
    let vx = pos_2[0] - pos_1[0];
    let vy = pos_2[1] - pos_1[1];
    let norm = (vx * vx + vy * vy).sqrt();
    if norm == 0.0 {
        return String::new();
    }
    let x2 = pos_2[0] - (vx / norm) * node_size;
    let y2 = pos_2[1] - (vy / norm) * node_size;
    let marker_id = marker_id_from_color(edge_color);
    format!(
        "<path stroke-width=\"{}\" stroke=\"{}\" d=\"M {} {} {} {}\" marker-end=\"url(#arrow-{})\"/>\n",
        edge_width, edge_color, pos_1[0] as i32, pos_1[1] as i32, x2 as i32, y2 as i32, marker_id
    )
}

fn circle_layout(n: usize, width: f64, height: f64, margin: f64) -> Vec<[f64; 2]> {
    if n == 0 {
        return Vec::new();
    }
    let cx = width / 2.0;
    let cy = height / 2.0;
    let r = (width.min(height) / 2.0 - margin).max(1.0);
    (0..n)
        .map(|i| {
            let theta = (i as f64) * std::f64::consts::TAU / (n as f64);
            [cx + r * theta.cos(), cy + r * theta.sin()]
        })
        .collect()
}

/// Computes visualize graph.
pub fn visualize_graph(
    adjacency: &CsMat<f64>,
    position: Option<&[[f64; 2]]>,
    names: Option<&[String]>,
    width: f64,
    height: f64,
    labels: Option<&[i32]>,
    scores: Option<&[f64]>,
    label_colors: Option<&[String]>,
    directed: bool,
    display_edges: bool,
    edge_width: f64,
    edge_color: &str,
    node_color: &str,
) -> String {
    let n = adjacency.rows();
    let pos = position
        .map(|p| p.to_vec())
        .unwrap_or_else(|| circle_layout(n, width, height, 20.0));
    let mut svg = String::with_capacity(256 + n * 128 + adjacency.nnz() * 96);
    let _ = write!(
        svg,
        "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
        width, height
    );
    if directed && display_edges {
        let edge_color_escaped = escape_xml_text(edge_color);
        let marker_id = marker_id_from_color(edge_color);
        let _ = write!(
            svg,
            "<defs><marker id=\"arrow-{}\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"{}\"/></marker></defs>\n",
            marker_id,
            edge_color_escaped
        );
    }
    if display_edges {
        for (i, row) in adjacency.outer_iterator().enumerate() {
            for (j, _v) in row.iter() {
                if directed {
                    svg.push_str(&svg_edge_directed(pos[i], pos[j], edge_width, edge_color, 6.0));
                } else {
                    svg.push_str(&svg_edge(pos[i], pos[j], edge_width, edge_color));
                }
            }
        }
    }
    let node_colors = get_node_colors(n, labels, scores, node_color, label_colors);
    for i in 0..n {
        svg.push_str(&svg_node(
            pos[i],
            6.0,
            node_colors.get(i).map(String::as_str).unwrap_or(node_color),
            1.0,
            "black",
        ));
    }
    if let Some(node_names) = names {
        for i in 0..n.min(node_names.len()) {
            svg.push_str(&svg_text(pos[i], &node_names[i], 8.0, 12, "right"));
        }
    }
    svg.push_str("</svg>\n");
    svg
}

/// Computes svg graph.
pub fn svg_graph(
    adjacency: &CsMat<f64>,
    position: Option<&[[f64; 2]]>,
    names: Option<&[String]>,
    width: f64,
    height: f64,
    labels: Option<&[i32]>,
    scores: Option<&[f64]>,
    label_colors: Option<&[String]>,
    directed: bool,
    display_edges: bool,
    edge_width: f64,
    edge_color: &str,
    node_color: &str,
) -> String {
    visualize_graph(
        adjacency,
        position,
        names,
        width,
        height,
        labels,
        scores,
        label_colors,
        directed,
        display_edges,
        edge_width,
        edge_color,
        node_color,
    )
}

/// Computes visualize bigraph.
pub fn visualize_bigraph(
    biadjacency: &CsMat<f64>,
    width: f64,
    height: f64,
    display_edges: bool,
    edge_width: f64,
    edge_color: &str,
) -> String {
    let (n_row, n_col) = biadjacency.shape();
    let mut pos_row = Vec::with_capacity(n_row);
    let mut pos_col = Vec::with_capacity(n_col);
    for i in 0..n_row {
        pos_row.push([width * 0.25, (i as f64 + 1.0) * height / (n_row as f64 + 1.0)]);
    }
    for j in 0..n_col {
        pos_col.push([width * 0.75, (j as f64 + 1.0) * height / (n_col as f64 + 1.0)]);
    }
    let mut svg = String::with_capacity(256 + (n_row + n_col) * 96 + biadjacency.nnz() * 80);
    let _ = write!(
        svg,
        "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
        width, height
    );
    if display_edges {
        for (i, row) in biadjacency.outer_iterator().enumerate() {
            for (j, _v) in row.iter() {
                svg.push_str(&svg_edge(pos_row[i], pos_col[j], edge_width, edge_color));
            }
        }
    }
    for p in pos_row {
        svg.push_str(&svg_node(p, 6.0, "gray", 1.0, "black"));
    }
    for p in pos_col {
        svg.push_str(&svg_node(p, 6.0, "gray", 1.0, "black"));
    }
    svg.push_str("</svg>\n");
    svg
}

/// Computes svg bigraph.
pub fn svg_bigraph(
    biadjacency: &CsMat<f64>,
    width: f64,
    height: f64,
    display_edges: bool,
    edge_width: f64,
    edge_color: &str,
) -> String {
    visualize_bigraph(biadjacency, width, height, display_edges, edge_width, edge_color)
}

/// Computes save svg.
pub fn save_svg(filename_without_ext: &str, svg: &str) -> std::io::Result<()> {
    let mut file = File::create(format!("{filename_without_ext}.svg"))?;
    file.write_all(svg.as_bytes())?;
    Ok(())
}

#[derive(Debug, Clone)]
/// GraphVizOptions value.
pub struct GraphVizOptions {
    /// Width value.
    pub width: f64,
    /// Height value.
    pub height: f64,
    /// Directed value.
    pub directed: bool,
    /// Display Edges value.
    pub display_edges: bool,
    /// Edge Width value.
    pub edge_width: f64,
    /// Edge Color value.
    pub edge_color: String,
    /// Node Color value.
    pub node_color: String,
}

impl Default for GraphVizOptions {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 300.0,
            directed: false,
            display_edges: true,
            edge_width: 1.0,
            edge_color: "gray".to_string(),
            node_color: "gray".to_string(),
        }
    }
}

#[derive(Debug, Clone)]
/// BigraphVizOptions value.
pub struct BigraphVizOptions {
    /// Width value.
    pub width: f64,
    /// Height value.
    pub height: f64,
    /// Display Edges value.
    pub display_edges: bool,
    /// Edge Width value.
    pub edge_width: f64,
    /// Edge Color value.
    pub edge_color: String,
}

impl Default for BigraphVizOptions {
    fn default() -> Self {
        Self {
            width: 400.0,
            height: 300.0,
            display_edges: true,
            edge_width: 1.0,
            edge_color: "gray".to_string(),
        }
    }
}

/// Computes visualize graph with options.
pub fn visualize_graph_with_options(
    adjacency: &CsMat<f64>,
    options: Option<GraphVizOptions>,
) -> String {
    let opts = options.unwrap_or_default();
    visualize_graph(
        adjacency,
        None,
        None,
        opts.width,
        opts.height,
        None,
        None,
        None,
        opts.directed,
        opts.display_edges,
        opts.edge_width,
        &opts.edge_color,
        &opts.node_color,
    )
}

/// Computes svg graph with options.
pub fn svg_graph_with_options(adjacency: &CsMat<f64>, options: Option<GraphVizOptions>) -> String {
    visualize_graph_with_options(adjacency, options)
}

/// Computes visualize bigraph with options.
pub fn visualize_bigraph_with_options(
    biadjacency: &CsMat<f64>,
    options: Option<BigraphVizOptions>,
) -> String {
    let opts = options.unwrap_or_default();
    visualize_bigraph(
        biadjacency,
        opts.width,
        opts.height,
        opts.display_edges,
        opts.edge_width,
        &opts.edge_color,
    )
}

/// Computes svg bigraph with options.
pub fn svg_bigraph_with_options(
    biadjacency: &CsMat<f64>,
    options: Option<BigraphVizOptions>,
) -> String {
    visualize_bigraph_with_options(biadjacency, options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::test_graphs::{test_bigraph, test_graph};

    #[test]
    fn test_svg_helpers() {
        assert_eq!(svg_text([0.0, 0.0], "foo", 1.0, 12, "right")[1..5].to_string(), "text");
        let escaped = svg_text([0.0, 0.0], "<a&\"b'>", 1.0, 12, "right");
        assert!(escaped.contains("&lt;a&amp;&quot;b&apos;&gt;"));
        assert!(!svg_node([0.0, 0.0], 2.0, "red", 1.0, "black").is_empty());
        assert!(!svg_edge([0.0, 0.0], [1.0, 1.0], 1.0, "black").is_empty());
        assert!(!svg_edge_directed([0.0, 0.0], [1.0, 1.0], 1.0, "black", 1.0).is_empty());
        let marker_id = marker_id_from_color("rgb(1, 2, 3)");
        assert!(marker_id.starts_with("rgb_1_2_3"));
    }

    #[test]
    fn test_visualize_graph_and_bigraph() {
        let g = test_graph();
        let image = visualize_graph(
            &g, None, None, 400.0, 300.0, None, None, None, false, true, 1.0, "gray", "gray",
        );
        assert_eq!(&image[1..4], "svg");
        let image = svg_graph(
            &g, None, None, 400.0, 300.0, None, None, None, true, true, 2.0, "blue", "red",
        );
        assert_eq!(&image[1..4], "svg");

        let b = test_bigraph();
        let image = visualize_bigraph(&b, 400.0, 300.0, true, 1.0, "gray");
        assert_eq!(&image[1..4], "svg");
        let image = svg_bigraph(&b, 400.0, 300.0, false, 1.0, "gray");
        assert_eq!(&image[1..4], "svg");
    }

    #[test]
    fn test_scaling_and_colors() {
        let x = min_max_scaling(&[1.0, 2.0, 3.0], None, None);
        assert_eq!(x.len(), 3);
        let colors = get_label_colors(None);
        assert!(!colors.is_empty());
    }

    #[test]
    fn test_write_svg() {
        let g = test_graph();
        let image = visualize_graph(
            &g, None, None, 200.0, 150.0, None, None, None, false, true, 1.0, "gray", "gray",
        );
        save_svg("stub_graph_image", &image).expect("save svg");
        let written = std::fs::read_to_string("stub_graph_image.svg").expect("read svg");
        assert_eq!(&written[1..4], "svg");
        let _ = std::fs::remove_file("stub_graph_image.svg");
    }

    #[test]
    fn test_node_color_modes() {
        let g = test_graph();
        let n = g.rows();
        let labels: Vec<i32> = (0..n).map(|i| (i % 3) as i32).collect();
        let scores: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let custom = vec!["red".to_string(), "blue".to_string(), "green".to_string()];

        let image_labels = visualize_graph(
            &g,
            None,
            None,
            300.0,
            200.0,
            Some(&labels),
            None,
            Some(&custom),
            false,
            true,
            1.0,
            "gray",
            "gray",
        );
        assert_eq!(&image_labels[1..4], "svg");

        let image_scores = visualize_graph(
            &g,
            None,
            None,
            300.0,
            200.0,
            None,
            Some(&scores),
            None,
            false,
            true,
            1.0,
            "gray",
            "gray",
        );
        assert_eq!(&image_scores[1..4], "svg");
    }

    #[test]
    fn test_options_wrappers() {
        let g = test_graph();
        let image = visualize_graph_with_options(&g, None);
        assert_eq!(&image[1..4], "svg");
        let image = svg_graph_with_options(&g, None);
        assert_eq!(&image[1..4], "svg");
        let image = visualize_graph_with_options(
            &g,
            Some(GraphVizOptions {
                directed: true,
                edge_color: "blue".to_string(),
                ..Default::default()
            }),
        );
        assert_eq!(&image[1..4], "svg");

        let b = test_bigraph();
        let image = visualize_bigraph_with_options(&b, None);
        assert_eq!(&image[1..4], "svg");
        let image = svg_bigraph_with_options(&b, None);
        assert_eq!(&image[1..4], "svg");
    }

    #[test]
    fn test_directed_marker_id_sanitized() {
        let g = test_graph();
        let image = visualize_graph(
            &g,
            None,
            None,
            300.0,
            200.0,
            None,
            None,
            None,
            true,
            true,
            1.0,
            "rgb(1, 2, 3)",
            "gray",
        );
        assert!(image.contains("id=\"arrow-rgb_1_2_3-"));
        assert!(image.contains("marker-end=\"url(#arrow-rgb_1_2_3-"));
    }
}
