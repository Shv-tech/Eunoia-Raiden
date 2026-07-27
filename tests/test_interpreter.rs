#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nibble_grid_packing() {
        // Guarantee our zero-allocation 450-byte grid accurately stores 4-bit values.
        let mut grid = Grid { data: [0; 450], rows: 10, cols: 10 };
        
        grid.set(5, 5, 9); // Max ARC color
        grid.set(5, 6, 3);
        
        assert_eq!(grid.get(5, 5), 9, "High-nibble packing failed.");
        assert_eq!(grid.get(5, 6), 3, "Low-nibble packing failed.");
        assert_eq!(grid.get(0, 0), 0, "Zero-constraint initialized incorrectly.");
    }
}