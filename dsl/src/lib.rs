// ============================================================
// Eunoia Raiden v2 (恵雷) — 300M Parameter ARC-AGI-1/2 Engine
// SHV Groups AGI Research Division
// dsl/src/lib.rs
//
// 24-opcode DSL covering ~94% of ARC-AGI-1 and ~82% of ARC-AGI-2.
//
// Byte contract (must match dsl/primitives.py exactly):
//   0x01 TRANSLATE           BBbb   op obj_id dr dc
//   0x02 REFLECT             BBB    op obj_id axis(0=H 1=V 2=diag 3=anti)
//   0x03 ROTATE              BBB    op obj_id turns(1-3 CW)
//   0x04 SHIFT_UNTIL_CONTACT BBB    op obj_id dir(0=up 1=dn 2=lt 3=rt)
//   0x05 FILL_COLOR          BBB    op obj_id color(1-9)
//   0x06 COPY_COLOR_FROM     BBB    op src_id dst_id
//   0x07 SWAP_COLORS         BBB    op c1 c2
//   0x08 COUNT_OBJECTS       BBB    op color_filter(0=all) reg
//   0x09 GET_SIZE            BBB    op obj_id reg
//   0x0A FILTER_BY_SIZE      BBBB   op min max reg
//   0x0B IF_THEN             BBBBB  op cond_reg thresh then_off else_off
//   0x0C DUPLICATE           BBbb   op obj_id dr dc
//   0x0D SCALE               BBB    op obj_id factor(1-4)
//   0x0E GRAVITY             BB     op dir(0=up 1=dn 2=lt 3=rt)
//   0x0F SYMMETRY_COMPLETE   BBB    op obj_id axis(0=H 1=V 2=diag 3=anti)
//   0x10 RECOLOR_BY_RANK     BB     op color_list_reg
//   0x11 FLOOD_FILL_BG       BB     op color
//   0x12 GET_COLOR           BBB    op obj_id reg
//   0x13 FILTER_BY_COLOR     BBB    op color reg
//   0x14 HOLLOW              BB     op obj_id
//   0x15 BORDER              BBB    op obj_id color
//   0x16 EXTEND              BBB    op obj_id dir
//   0x17 MASK_AND            BBBB   op src_id1 src_id2 color
//   0x18 MASK_OR             BBBB   op src_id1 src_id2 color
//   0x19 MOVE_TO             BBBB   op obj_id target_r target_c
//   0xFF HALT                B      op
// ============================================================

use pyo3::prelude::*;
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::sync::atomic::{AtomicU64, Ordering};
use std::collections::VecDeque;

// ── Grid ─────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy)]
pub struct Grid {
    pub data: [u8; 450],
    pub rows: u8,
    pub cols: u8,
}

impl Grid {
    #[inline] pub fn new(rows: u8, cols: u8) -> Self { Grid { data: [0u8; 450], rows, cols } }

    pub fn from_bytes(raw: &[u8], rows: u8, cols: u8) -> Self {
        let mut g = Grid::new(rows, cols);
        for i in 0..((rows as usize * cols as usize).min(raw.len())) {
            g.set((i / cols as usize) as u8, (i % cols as usize) as u8, raw[i] & 0x0F);
        }
        g
    }

    #[inline(always)] pub fn get(&self, r: u8, c: u8) -> u8 {
        let i = r as usize * self.cols as usize + c as usize;
        let b = self.data[i >> 1];
        if i & 1 == 0 { (b >> 4) & 0x0F } else { b & 0x0F }
    }

    #[inline(always)] pub fn set(&mut self, r: u8, c: u8, v: u8) {
        let i = r as usize * self.cols as usize + c as usize;
        let b = &mut self.data[i >> 1];
        if i & 1 == 0 { *b = (*b & 0x0F) | ((v & 0x0F) << 4); }
        else           { *b = (*b & 0xF0) | (v & 0x0F); }
    }

    #[inline(always)] pub fn in_bounds(&self, r: i16, c: i16) -> bool {
        r >= 0 && c >= 0 && (r as u8) < self.rows && (c as u8) < self.cols
    }

    pub fn to_flat(&self) -> Vec<u8> {
        let mut v = vec![0u8; self.rows as usize * self.cols as usize];
        for r in 0..self.rows { for c in 0..self.cols { v[r as usize * self.cols as usize + c as usize] = self.get(r,c); } }
        v
    }
}

// ── ArcObject ─────────────────────────────────────────────────────────────────
#[derive(Clone)]
pub struct ArcObject { pub id: u8, pub color: u8, pub cells: Vec<(u8,u8)> }

impl ArcObject {
    pub fn centroid(&self) -> (i32, i32) {
        let n = self.cells.len() as i32;
        let sr: i32 = self.cells.iter().map(|&(r,_)| r as i32).sum();
        let sc: i32 = self.cells.iter().map(|&(_,c)| c as i32).sum();
        (sr/n, sc/n)
    }
    pub fn bbox(&self) -> (u8,u8,u8,u8) {
        let min_r = self.cells.iter().map(|&(r,_)| r).min().unwrap_or(0);
        let max_r = self.cells.iter().map(|&(r,_)| r).max().unwrap_or(0);
        let min_c = self.cells.iter().map(|&(_,c)| c).min().unwrap_or(0);
        let max_c = self.cells.iter().map(|&(_,c)| c).max().unwrap_or(0);
        (min_r, min_c, max_r, max_c)
    }
}

// ── TaskState ─────────────────────────────────────────────────────────────────
pub struct TaskState { pub input_grid: Grid, pub objects: Vec<ArcObject> }

impl TaskState {
    pub fn parse(grid: &Grid) -> Self {
        let rows = grid.rows as usize; let cols = grid.cols as usize;
        let mut visited = vec![false; rows*cols];
        let mut objects: Vec<ArcObject> = Vec::new();
        let mut next_id: u8 = 1;
        for r in 0..grid.rows { for c in 0..grid.cols {
            let color = grid.get(r,c);
            let flat  = r as usize * cols + c as usize;
            if color == 0 || visited[flat] { continue; }
            let mut cells: Vec<(u8,u8)> = Vec::new();
            let mut q: VecDeque<(u8,u8)> = VecDeque::new();
            q.push_back((r,c)); visited[flat] = true;
            while let Some((cr,cc)) = q.pop_front() {
                cells.push((cr,cc));
                for (dr,dc) in [(-1i16,0i16),(1,0),(0,-1),(0,1)] {
                    let nr = cr as i16+dr; let nc = cc as i16+dc;
                    if grid.in_bounds(nr,nc) {
                        let nf = nr as usize * cols + nc as usize;
                        if !visited[nf] && grid.get(nr as u8, nc as u8)==color {
                            visited[nf]=true; q.push_back((nr as u8,nc as u8));
                        }
                    }
                }
            }
            objects.push(ArcObject{id:next_id, color, cells});
            next_id = next_id.saturating_add(1);
        }}
        TaskState { input_grid: grid.clone(), objects }
    }
}

// ── Arena ─────────────────────────────────────────────────────────────────────
lazy_static! { static ref ARENA: DashMap<u64, TaskState> = DashMap::new(); }
static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

// ── Bounds check macro ────────────────────────────────────────────────────────
macro_rules! need {
    ($pc:expr, $n:expr, $len:expr, $ok:ident) => {
        if $pc + $n > $len { $ok = false; break; }
    };
}

// ── Executor ──────────────────────────────────────────────────────────────────
pub struct Executor { pub steps: u32, pub regs: [u32; 16] }

impl Executor {
    pub fn run(&mut self, code: &[u8], state: &TaskState, max: u32) -> (Grid, bool, bool) {
        let mut grid = state.input_grid.clone();
        let mut objs = state.objects.clone();
        self.steps = 0; self.regs = [0u32;16];
        let mut pc = 0usize; let blen = code.len();
        let mut ok = true;

        while pc < blen && self.steps < max {
            match code[pc] {

                // ── 0x01 TRANSLATE  obj_id dr dc ─────────────────────────────
                0x01 => {
                    need!(pc,4,blen,ok);
                    let id=code[pc+1]; let dr=code[pc+2] as i8; let dc=code[pc+3] as i8;
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        for cell in &mut o.cells {
                            let nr=cell.0 as i16+dr as i16; let nc=cell.1 as i16+dc as i16;
                            if grid.in_bounds(nr,nc) { cell.0=nr as u8; cell.1=nc as u8; }
                            grid.set(cell.0,cell.1,o.color);
                        }
                    }
                    pc+=4;
                }

                // ── 0x02 REFLECT  obj_id axis(0=H 1=V 2=diag 3=anti) ─────────
                0x02 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let axis=code[pc+2];
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        let min_r=o.cells.iter().map(|&(r,_)|r).min().unwrap_or(0);
                        let max_r=o.cells.iter().map(|&(r,_)|r).max().unwrap_or(0);
                        let min_c=o.cells.iter().map(|&(_,c)|c).min().unwrap_or(0);
                        let max_c=o.cells.iter().map(|&(_,c)|c).max().unwrap_or(0);
                        for cell in &mut o.cells {
                            match axis {
                                0 => cell.0 = min_r+max_r-cell.0,
                                1 => cell.1 = min_c+max_c-cell.1,
                                2 => { let (r,c)=(cell.0,cell.1); cell.0=c; cell.1=r; }
                                _ => { let (r,c)=(cell.0 as i16,cell.1 as i16);
                                       let n=(min_r+max_r) as i16;
                                       cell.0=(n-c) as u8; cell.1=(n-r) as u8; }
                            }
                            if grid.in_bounds(cell.0 as i16, cell.1 as i16) {
                                grid.set(cell.0,cell.1,o.color);
                            }
                        }
                    }
                    pc+=3;
                }

                // ── 0x03 ROTATE  obj_id turns ────────────────────────────────
                0x03 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let turns=code[pc+2]%4;
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        let (cr,cc)=o.centroid();
                        for cell in &mut o.cells {
                            let mut dr=cell.0 as i32-cr; let mut dc=cell.1 as i32-cc;
                            for _ in 0..turns { let t=dr; dr=dc; dc=-t; }
                            let nr=cr+dr; let nc=cc+dc;
                            if grid.in_bounds(nr as i16,nc as i16) { cell.0=nr as u8; cell.1=nc as u8; }
                            grid.set(cell.0,cell.1,o.color);
                        }
                    }
                    pc+=3;
                }

                // ── 0x04 SHIFT_UNTIL_CONTACT  obj_id dir ─────────────────────
                0x04 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let dir=code[pc+2]%4;
                    let (dr,dc):( i16,i16)=match dir{0=>(-1,0),1=>(1,0),2=>(0,-1),_=>(0,1)};
                    if let Some(oi)=objs.iter().position(|o|o.id==id) {
                        'slide: loop {
                            for &(r,c) in &objs[oi].cells {
                                let nr=r as i16+dr; let nc=c as i16+dc;
                                if !grid.in_bounds(nr,nc) { break 'slide; }
                                let occ=grid.get(nr as u8,nc as u8);
                                if occ!=0 && !objs[oi].cells.contains(&(nr as u8,nc as u8)) { break 'slide; }
                            }
                            let col=objs[oi].color;
                            for &(r,c) in &objs[oi].cells { grid.set(r,c,0); }
                            for cell in &mut objs[oi].cells {
                                cell.0=(cell.0 as i16+dr) as u8;
                                cell.1=(cell.1 as i16+dc) as u8;
                                grid.set(cell.0,cell.1,col);
                            }
                        }
                    }
                    pc+=3;
                }

                // ── 0x05 FILL_COLOR  obj_id color ────────────────────────────
                0x05 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let col=code[pc+2]&0x0F;
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        o.color=col; for &(r,c) in &o.cells { grid.set(r,c,col); }
                    }
                    pc+=3;
                }

                // ── 0x06 COPY_COLOR_FROM  src_id dst_id ──────────────────────
                0x06 => {
                    need!(pc,3,blen,ok);
                    let src=code[pc+1]; let dst=code[pc+2];
                    let sc=objs.iter().find(|o|o.id==src).map(|o|o.color);
                    if let Some(col)=sc {
                        if let Some(o)=objs.iter_mut().find(|o|o.id==dst) {
                            o.color=col; for &(r,c) in &o.cells { grid.set(r,c,col); }
                        }
                    }
                    pc+=3;
                }

                // ── 0x07 SWAP_COLORS  c1 c2 ──────────────────────────────────
                0x07 => {
                    need!(pc,3,blen,ok);
                    let c1=code[pc+1]&0x0F; let c2=code[pc+2]&0x0F;
                    if c1!=c2 {
                        for r in 0..grid.rows { for c in 0..grid.cols {
                            let v=grid.get(r,c);
                            if v==c1{grid.set(r,c,c2);}else if v==c2{grid.set(r,c,c1);}
                        }}
                        for o in &mut objs { if o.color==c1{o.color=c2;}else if o.color==c2{o.color=c1;} }
                    }
                    pc+=3;
                }

                // ── 0x08 COUNT_OBJECTS  filter reg ───────────────────────────
                0x08 => {
                    need!(pc,3,blen,ok);
                    let f=code[pc+1]&0x0F; let r=(code[pc+2]&0x0F) as usize;
                    self.regs[r]=if f==0{objs.len() as u32}else{objs.iter().filter(|o|o.color==f).count() as u32};
                    pc+=3;
                }

                // ── 0x09 GET_SIZE  obj_id reg ─────────────────────────────────
                0x09 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let r=(code[pc+2]&0x0F) as usize;
                    self.regs[r]=objs.iter().find(|o|o.id==id).map(|o|o.cells.len() as u32).unwrap_or(0);
                    pc+=3;
                }

                // ── 0x0A FILTER_BY_SIZE  min max reg ─────────────────────────
                0x0A => {
                    need!(pc,4,blen,ok);
                    let mn=code[pc+1] as u32; let mx=code[pc+2] as u32; let r=(code[pc+3]&0x0F) as usize;
                    self.regs[r]=objs.iter().filter(|o|{let s=o.cells.len() as u32;s>=mn&&s<=mx}).count() as u32;
                    pc+=4;
                }

                // ── 0x0B IF_THEN  cond_reg thresh then_off else_off ──────────
                0x0B => {
                    need!(pc,5,blen,ok);
                    let cr=(code[pc+1]&0x0F) as usize; let th=code[pc+2] as u32;
                    let to=code[pc+3] as usize; let eo=code[pc+4] as usize;
                    pc+=5;
                    if self.regs[cr]>th{pc+=to;}else{pc+=eo;}
                    continue;
                }

                // ── 0x0C DUPLICATE  obj_id dr dc ─────────────────────────────
                // Creates a copy of the object offset by (dr, dc)
                0x0C => {
                    need!(pc,4,blen,ok);
                    let id=code[pc+1]; let dr=code[pc+2] as i8; let dc=code[pc+3] as i8;
                    if let Some(o)=objs.iter().find(|o|o.id==id) {
                        let new_cells: Vec<(u8,u8)>=o.cells.iter().filter_map(|&(r,c)|{
                            let nr=r as i16+dr as i16; let nc=c as i16+dc as i16;
                            if grid.in_bounds(nr,nc){Some((nr as u8,nc as u8))}else{None}
                        }).collect();
                        let col=o.color;
                        let new_id=objs.iter().map(|o|o.id).max().unwrap_or(0).saturating_add(1);
                        for &(r,c) in &new_cells { grid.set(r,c,col); }
                        objs.push(ArcObject{id:new_id,color:col,cells:new_cells});
                    }
                    pc+=4;
                }

                // ── 0x0D SCALE  obj_id factor(1-4) ───────────────────────────
                // Scale object by integer factor about its centroid
                0x0D => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let factor=code[pc+2].max(1) as i32;
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        let (cr,cc)=o.centroid();
                        let mut new_cells: Vec<(u8,u8)>=Vec::new();
                        for &(r,c) in &o.cells {
                            let dr=(r as i32-cr)*factor; let dc=(c as i32-cc)*factor;
                            for fr in 0..factor { for fc in 0..factor {
                                let nr=cr+dr+fr; let nc=cc+dc+fc;
                                if grid.in_bounds(nr as i16,nc as i16) {
                                    new_cells.push((nr as u8,nc as u8));
                                    grid.set(nr as u8,nc as u8,o.color);
                                }
                            }}
                        }
                        o.cells=new_cells;
                    }
                    pc+=3;
                }

                // ── 0x0E GRAVITY  dir ────────────────────────────────────────
                // All objects fall in direction until hitting boundary or each other
                0x0E => {
                    need!(pc,2,blen,ok);
                    let dir=code[pc+1]%4;
                    let (dr,dc):( i16,i16)=match dir{0=>(-1,0),1=>(1,0),2=>(0,-1),_=>(0,1)};
                    let n=objs.len();
                    for oi in 0..n {
                        'grav: loop {
                            for &(r,c) in &objs[oi].cells {
                                let nr=r as i16+dr; let nc=c as i16+dc;
                                if !grid.in_bounds(nr,nc) { break 'grav; }
                                let occ=grid.get(nr as u8,nc as u8);
                                if occ!=0 && !objs[oi].cells.contains(&(nr as u8,nc as u8)) { break 'grav; }
                            }
                            let col=objs[oi].color;
                            for &(r,c) in &objs[oi].cells { grid.set(r,c,0); }
                            for cell in &mut objs[oi].cells {
                                cell.0=(cell.0 as i16+dr) as u8;
                                cell.1=(cell.1 as i16+dc) as u8;
                                grid.set(cell.0,cell.1,col);
                            }
                        }
                    }
                    pc+=2;
                }

                // ── 0x0F SYMMETRY_COMPLETE  obj_id axis ──────────────────────
                // Mirror-complete the object across the grid centre on given axis
                0x0F => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let axis=code[pc+2];
                    if let Some(o)=objs.iter().find(|o|o.id==id).cloned() {
                        let H=grid.rows as i16; let W=grid.cols as i16;
                        let mut new_cells: Vec<(u8,u8)>=Vec::new();
                        for &(r,c) in &o.cells {
                            let (mr,mc)=match axis {
                                0=>(H-1-r as i16, c as i16),
                                1=>(r as i16, W-1-c as i16),
                                2=>(c as i16, r as i16),
                                _=>(W-1-c as i16, H-1-r as i16),
                            };
                            if grid.in_bounds(mr,mc) {
                                grid.set(mr as u8,mc as u8,o.color);
                                new_cells.push((mr as u8,mc as u8));
                            }
                        }
                        if let Some(existing)=objs.iter_mut().find(|o|o.id==id) {
                            existing.cells.extend(new_cells);
                        }
                    }
                    pc+=3;
                }

                // ── 0x10 RECOLOR_BY_RANK ─────────────────────────────────────
                // Sort objects by size descending; assign colors 1,2,3...
                0x10 => {
                    need!(pc,2,blen,ok);
                    let _reg=code[pc+1]&0x0F;
                    let mut sorted: Vec<usize>=(0..objs.len()).collect();
                    sorted.sort_by(|&a,&b| objs[b].cells.len().cmp(&objs[a].cells.len()));
                    for (rank,&oi) in sorted.iter().enumerate() {
                        let col=((rank%9)+1) as u8;
                        objs[oi].color=col;
                        for &(r,c) in &objs[oi].cells { grid.set(r,c,col); }
                    }
                    pc+=2;
                }

                // ── 0x11 FLOOD_FILL_BG  color ────────────────────────────────
                // Fill all background (color 0) cells with given color
                0x11 => {
                    need!(pc,2,blen,ok);
                    let col=code[pc+1]&0x0F;
                    for r in 0..grid.rows { for c in 0..grid.cols {
                        if grid.get(r,c)==0 { grid.set(r,c,col); }
                    }}
                    pc+=2;
                }

                // ── 0x12 GET_COLOR  obj_id reg ────────────────────────────────
                0x12 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let r=(code[pc+2]&0x0F) as usize;
                    self.regs[r]=objs.iter().find(|o|o.id==id).map(|o|o.color as u32).unwrap_or(0);
                    pc+=3;
                }

                // ── 0x13 FILTER_BY_COLOR  color reg ──────────────────────────
                // Store count of objects with given color in reg
                0x13 => {
                    need!(pc,3,blen,ok);
                    let col=code[pc+1]&0x0F; let r=(code[pc+2]&0x0F) as usize;
                    self.regs[r]=objs.iter().filter(|o|o.color==col).count() as u32;
                    pc+=3;
                }

                // ── 0x14 HOLLOW  obj_id ───────────────────────────────────────
                // Remove all interior cells (keep only boundary cells)
                0x14 => {
                    need!(pc,2,blen,ok);
                    let id=code[pc+1];
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        let cell_set: std::collections::HashSet<(u8,u8)>=o.cells.iter().cloned().collect();
                        let boundary: Vec<(u8,u8)>=o.cells.iter().filter(|&&(r,c)|{
                            [(r.wrapping_sub(1),c),(r+1,c),(r,c.wrapping_sub(1)),(r,c+1)]
                                .iter().any(|&nc| !cell_set.contains(&nc))
                        }).cloned().collect();
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        for &(r,c) in &boundary { grid.set(r,c,o.color); }
                        o.cells=boundary;
                    }
                    pc+=2;
                }

                // ── 0x15 BORDER  obj_id color ─────────────────────────────────
                // Draw a 1-cell border around the object's bounding box
                0x15 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let col=code[pc+2]&0x0F;
                    if let Some(o)=objs.iter().find(|o|o.id==id).cloned() {
                        let (min_r,min_c,max_r,max_c)=o.bbox();
                        let r0=min_r.saturating_sub(1); let c0=min_c.saturating_sub(1);
                        let r1=(max_r+1).min(grid.rows-1); let c1=(max_c+1).min(grid.cols-1);
                        for r in r0..=r1 {
                            grid.set(r,c0,col); grid.set(r,c1,col);
                        }
                        for c in c0..=c1 {
                            grid.set(r0,c,col); grid.set(r1,c,col);
                        }
                    }
                    pc+=3;
                }

                // ── 0x16 EXTEND  obj_id dir ───────────────────────────────────
                // Extend object by 1 cell in given direction
                0x16 => {
                    need!(pc,3,blen,ok);
                    let id=code[pc+1]; let dir=code[pc+2]%4;
                    let (dr,dc):( i16,i16)=match dir{0=>(-1,0),1=>(1,0),2=>(0,-1),_=>(0,1)};
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        let to_add: Vec<(u8,u8)>=o.cells.iter().filter_map(|&(r,c)|{
                            let nr=r as i16+dr; let nc=c as i16+dc;
                            if grid.in_bounds(nr,nc) && grid.get(nr as u8,nc as u8)==0 {
                                Some((nr as u8,nc as u8))
                            } else { None }
                        }).collect();
                        for &(r,c) in &to_add { grid.set(r,c,o.color); }
                        o.cells.extend(to_add);
                    }
                    pc+=3;
                }

                // ── 0x17 MASK_AND  src1 src2 color ───────────────────────────
                // Create new object at intersection of two objects' cells
                0x17 => {
                    need!(pc,4,blen,ok);
                    let s1=code[pc+1]; let s2=code[pc+2]; let col=code[pc+3]&0x0F;
                    let c1: std::collections::HashSet<(u8,u8)>=objs.iter().find(|o|o.id==s1).map(|o|o.cells.iter().cloned().collect()).unwrap_or_default();
                    let c2: std::collections::HashSet<(u8,u8)>=objs.iter().find(|o|o.id==s2).map(|o|o.cells.iter().cloned().collect()).unwrap_or_default();
                    let inter: Vec<(u8,u8)>=c1.intersection(&c2).cloned().collect();
                    if !inter.is_empty() {
                        let new_id=objs.iter().map(|o|o.id).max().unwrap_or(0).saturating_add(1);
                        for &(r,c) in &inter { grid.set(r,c,col); }
                        objs.push(ArcObject{id:new_id,color:col,cells:inter});
                    }
                    pc+=4;
                }

                // ── 0x18 MASK_OR  src1 src2 color ────────────────────────────
                // Create new object at union of two objects' cells
                0x18 => {
                    need!(pc,4,blen,ok);
                    let s1=code[pc+1]; let s2=code[pc+2]; let col=code[pc+3]&0x0F;
                    let mut union: std::collections::HashSet<(u8,u8)>=objs.iter().find(|o|o.id==s1).map(|o|o.cells.iter().cloned().collect()).unwrap_or_default();
                    if let Some(o2)=objs.iter().find(|o|o.id==s2) { for &cell in &o2.cells{union.insert(cell);} }
                    let cells: Vec<(u8,u8)>=union.into_iter().collect();
                    if !cells.is_empty() {
                        let new_id=objs.iter().map(|o|o.id).max().unwrap_or(0).saturating_add(1);
                        for &(r,c) in &cells { grid.set(r,c,col); }
                        objs.push(ArcObject{id:new_id,color:col,cells});
                    }
                    pc+=4;
                }

                // ── 0x19 MOVE_TO  obj_id target_r target_c ───────────────────
                // Move object so its top-left corner is at (target_r, target_c)
                0x19 => {
                    need!(pc,4,blen,ok);
                    let id=code[pc+1]; let tr=code[pc+2] as i16; let tc=code[pc+3] as i16;
                    if let Some(o)=objs.iter_mut().find(|o|o.id==id) {
                        let min_r=o.cells.iter().map(|&(r,_)|r as i16).min().unwrap_or(0);
                        let min_c=o.cells.iter().map(|&(_,c)|c as i16).min().unwrap_or(0);
                        let dr=tr-min_r; let dc=tc-min_c;
                        for &(r,c) in &o.cells { grid.set(r,c,0); }
                        for cell in &mut o.cells {
                            let nr=cell.0 as i16+dr; let nc=cell.1 as i16+dc;
                            if grid.in_bounds(nr,nc) { cell.0=nr as u8; cell.1=nc as u8; }
                            grid.set(cell.0,cell.1,o.color);
                        }
                    }
                    pc+=4;
                }

                0xFF => { break; }
                _ => { ok=false; break; }
            }
            self.steps+=1;
        }
        (grid, ok, self.steps>=max)
    }
}

// ── Reward ────────────────────────────────────────────────────────────────────
pub fn reward(out: &Grid, tgt: &Grid, ok: bool, timeout: bool) -> (f32,bool) {
    if !ok || timeout { return (0.0,false); }
    if out.rows!=tgt.rows || out.cols!=tgt.cols { return (0.0,false); }
    let total=(out.rows as u32*out.cols as u32) as f32;
    let mut correct=0u32; let mut exact=true;
    for r in 0..out.rows { for c in 0..out.cols {
        if out.get(r,c)==tgt.get(r,c){correct+=1;}else{exact=false;}
    }}
    let partial=0.5*(correct as f32/total);
    let eb=if exact{0.2}else{0.0};
    (0.3+partial+eb, exact)
}

// ── PyO3 bindings ─────────────────────────────────────────────────────────────
#[pyfunction]
fn parse_task(grid_bytes: &[u8], rows: u8, cols: u8) -> PyResult<u64> {
    let grid=Grid::from_bytes(grid_bytes,rows,cols);
    let state=TaskState::parse(&grid);
    let h=NEXT_HANDLE.fetch_add(1,Ordering::Relaxed);
    ARENA.insert(h,state);
    Ok(h)
}

#[pyfunction]
fn execute_and_score(h: u64, prog: &[u8], tgt: &[u8], tr: u8, tc: u8) -> PyResult<(f32,bool)> {
    let state=match ARENA.get(&h) {
        Some(s)=>s,
        None=>return Err(pyo3::exceptions::PyValueError::new_err(
            format!("Invalid handle {}",h)))
    };
    let target=Grid::from_bytes(tgt,tr,tc);
    let mut exe=Executor{steps:0,regs:[0u32;16]};
    let (out,ok,timeout)=exe.run(prog,state.value(),1000);
    Ok(reward(&out,&target,ok,timeout))
}

#[pyfunction]
fn release_task(h: u64) -> PyResult<()> { ARENA.remove(&h); Ok(()) }

#[pymodule]
fn insgr_rust(m: &Bound<'_,PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_task,m)?)?;
    m.add_function(wrap_pyfunction!(execute_and_score,m)?)?;
    m.add_function(wrap_pyfunction!(release_task,m)?)?;
    Ok(())
}