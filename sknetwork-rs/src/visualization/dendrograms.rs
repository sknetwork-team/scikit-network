use crate::hierarchy::postprocess::Dendrogram;
use crate::hierarchy::postprocess::cut_straight;
use crate::visualization::colors::STANDARD_COLORS;
use std::fmt::Write as FmtWrite;

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

fn get_index(dendrogram: &Dendrogram, reorder: bool) -> Vec<usize> {
    let n = dendrogram.len() + 1;
    let mut tree: std::collections::HashMap<usize, Vec<usize>> = (0..n).map(|i| (i, vec![i])).collect();
    for (t, row) in dendrogram.iter().enumerate().take(n - 1) {
        let i = row[0] as usize;
        let j = row[1] as usize;
        let left = tree.remove(&i).unwrap_or_default();
        let right = tree.remove(&j).unwrap_or_default();
        let merged = if reorder && left.len() < right.len() {
            right.into_iter().chain(left.into_iter()).collect()
        } else {
            left.into_iter().chain(right.into_iter()).collect()
        };
        tree.insert(n + t, merged);
    }
    tree.into_values().next().unwrap_or_default()
}

/// Computes svg dendrogram top.
pub fn svg_dendrogram_top(
    dendrogram: &Dendrogram,
    names: Option<&[String]>,
    width: f64,
    height: f64,
    n_clusters: usize,
    reorder: bool,
) -> String {
    if dendrogram.is_empty() {
        return "<svg></svg>".to_string();
    }
    let labels = cut_straight(dendrogram, Some(n_clusters), None, true).unwrap_or_default();
    let colors: Vec<String> = STANDARD_COLORS.iter().map(|s| s.to_string()).collect();
    let index = get_index(dendrogram, reorder);
    let n = index.len();
    let h_max = dendrogram.last().map(|r| r[2]).unwrap_or(1.0).max(1.0);
    let unit_height = height / h_max;
    let unit_width = width / (n as f64).max(1.0);
    let mut pos: std::collections::HashMap<usize, (f64, f64)> = (0..n)
        .map(|k| (index[k], ((k as f64) * unit_width + 10.0, height + 10.0)))
        .collect();
    let mut label_map: std::collections::HashMap<usize, usize> =
        (0..labels.len()).map(|i| (i, labels[i])).collect();
    let mut svg = String::with_capacity(256 + n * 96);
    let _ = write!(
        svg,
        "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">",
        width + 20.0,
        height + 40.0
    );
    for (t, row) in dendrogram.iter().enumerate().take(n - 1) {
        let i = row[0] as usize;
        let j = row[1] as usize;
        let (x1, y1) = pos.remove(&i).unwrap_or((0.0, 0.0));
        let (x2, y2) = pos.remove(&j).unwrap_or((0.0, 0.0));
        let l1 = label_map.remove(&i).unwrap_or(0);
        let l2 = label_map.remove(&j).unwrap_or(0);
        let color = if l1 == l2 {
            &colors[l1 % colors.len()]
        } else {
            "black"
        };
        let x = 0.5 * (x1 + x2);
        let y = height + 10.0 - row[2] * unit_height;
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x1 as i32, y1 as i32, x1 as i32, y as i32));
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x2 as i32, y2 as i32, x2 as i32, y as i32));
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x1 as i32, y as i32, x2 as i32, y as i32));
        pos.insert(n + t, (x, y));
        label_map.insert(n + t, l1);
    }
    if let Some(node_names) = names {
        for i in 0..n.min(node_names.len()) {
            let _ = write!(
                svg,
                "<text x=\"{}\" y=\"{}\" font-size=\"12\">{}</text>",
                (i as f64 * unit_width + 5.0) as i32,
                (height + 25.0) as i32,
                escape_xml_text(&node_names[i])
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

/// Computes svg dendrogram left.
pub fn svg_dendrogram_left(
    dendrogram: &Dendrogram,
    names: Option<&[String]>,
    width: f64,
    height: f64,
    n_clusters: usize,
    reorder: bool,
) -> String {
    if dendrogram.is_empty() {
        return "<svg></svg>".to_string();
    }
    let labels = cut_straight(dendrogram, Some(n_clusters), None, true).unwrap_or_default();
    let colors: Vec<String> = STANDARD_COLORS.iter().map(|s| s.to_string()).collect();
    let index = get_index(dendrogram, reorder);
    let n = index.len();
    let h_max = dendrogram.last().map(|r| r[2]).unwrap_or(1.0).max(1.0);
    let unit_height = height / (n as f64).max(1.0);
    let unit_width = width / h_max;
    let mut pos: std::collections::HashMap<usize, (f64, f64)> = (0..n)
        .map(|k| (index[k], (width + 10.0, (k as f64) * unit_height + 10.0)))
        .collect();
    let mut label_map: std::collections::HashMap<usize, usize> =
        (0..labels.len()).map(|i| (i, labels[i])).collect();
    let mut svg = String::with_capacity(256 + n * 96);
    let _ = write!(
        svg,
        "<svg width=\"{}\" height=\"{}\" xmlns=\"http://www.w3.org/2000/svg\">",
        width + 60.0,
        height + 20.0
    );
    for (t, row) in dendrogram.iter().enumerate().take(n - 1) {
        let i = row[0] as usize;
        let j = row[1] as usize;
        let (x1, y1) = pos.remove(&i).unwrap_or((0.0, 0.0));
        let (x2, y2) = pos.remove(&j).unwrap_or((0.0, 0.0));
        let l1 = label_map.remove(&i).unwrap_or(0);
        let l2 = label_map.remove(&j).unwrap_or(0);
        let color = if l1 == l2 {
            &colors[l1 % colors.len()]
        } else {
            "black"
        };
        let y = 0.5 * (y1 + y2);
        let x = width + 10.0 - row[2] * unit_width;
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x1 as i32, y1 as i32, x as i32, y1 as i32));
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x2 as i32, y2 as i32, x as i32, y2 as i32));
        svg.push_str(&format!("<path stroke-width=\"1\" stroke=\"{}\" d=\"M {} {} {} {}\" />", color, x as i32, y1 as i32, x as i32, y2 as i32));
        pos.insert(n + t, (x, y));
        label_map.insert(n + t, l1);
    }
    if let Some(node_names) = names {
        for i in 0..n.min(node_names.len()) {
            let _ = write!(
                svg,
                "<text x=\"{}\" y=\"{}\" font-size=\"12\">{}</text>",
                (width + 15.0) as i32,
                ((i as f64) * unit_height + 14.0) as i32,
                escape_xml_text(&node_names[i])
            );
        }
    }
    svg.push_str("</svg>");
    svg
}

/// Computes visualize dendrogram.
pub fn visualize_dendrogram(
    dendrogram: &Dendrogram,
    names: Option<&[String]>,
    rotate: bool,
    width: f64,
    height: f64,
    n_clusters: usize,
    reorder: bool,
) -> String {
    if rotate {
        svg_dendrogram_left(dendrogram, names, width, height, n_clusters, reorder)
    } else {
        svg_dendrogram_top(dendrogram, names, width, height, n_clusters, reorder)
    }
}

/// Computes svg dendrogram.
pub fn svg_dendrogram(
    dendrogram: &Dendrogram,
    names: Option<&[String]>,
    rotate: bool,
    width: f64,
    height: f64,
    n_clusters: usize,
    reorder: bool,
) -> String {
    visualize_dendrogram(dendrogram, names, rotate, width, height, n_clusters, reorder)
}

#[derive(Debug, Clone)]
/// DendrogramVizOptions value.
pub struct DendrogramVizOptions {
    /// Rotate value.
    pub rotate: bool,
    /// Width value.
    pub width: f64,
    /// Height value.
    pub height: f64,
    /// N Clusters value.
    pub n_clusters: usize,
    /// Reorder value.
    pub reorder: bool,
}

impl Default for DendrogramVizOptions {
    fn default() -> Self {
        Self {
            rotate: false,
            width: 400.0,
            height: 300.0,
            n_clusters: 2,
            reorder: false,
        }
    }
}

/// Computes visualize dendrogram with options.
pub fn visualize_dendrogram_with_options(
    dendrogram: &Dendrogram,
    options: Option<DendrogramVizOptions>,
) -> String {
    let opts = options.unwrap_or_default();
    visualize_dendrogram(
        dendrogram,
        None,
        opts.rotate,
        opts.width,
        opts.height,
        opts.n_clusters,
        opts.reorder,
    )
}

/// Computes svg dendrogram with options.
pub fn svg_dendrogram_with_options(
    dendrogram: &Dendrogram,
    options: Option<DendrogramVizOptions>,
) -> String {
    visualize_dendrogram_with_options(dendrogram, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualize_dendrogram() {
        let d = vec![
            [0.0, 1.0, 1.0, 2.0],
            [2.0, 3.0, 2.0, 3.0],
        ];
        let image = visualize_dendrogram(&d, None, false, 400.0, 300.0, 2, false);
        assert_eq!(&image[1..4], "svg");
        let image = svg_dendrogram(&d, None, true, 400.0, 300.0, 2, true);
        assert_eq!(&image[1..4], "svg");
        let image = svg_dendrogram_top(&d, None, 400.0, 300.0, 2, false);
        assert_eq!(&image[1..4], "svg");
        let image = svg_dendrogram_left(&d, None, 400.0, 300.0, 2, false);
        assert_eq!(&image[1..4], "svg");
        let image = visualize_dendrogram_with_options(&d, None);
        assert_eq!(&image[1..4], "svg");
        let image = svg_dendrogram_with_options(&d, None);
        assert_eq!(&image[1..4], "svg");
        let names = vec!["<a&\"b'>".to_string(), "n2".to_string(), "n3".to_string()];
        let image = svg_dendrogram_top(&d, Some(&names), 400.0, 300.0, 2, false);
        assert!(image.contains("&lt;a&amp;&quot;b&apos;&gt;"));
    }
}
