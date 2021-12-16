use std::str::Bytes;

const PUZZLE_INPUT: &str = "A20D6CE8F00033925A95338B6549C0149E3398DE75817200992531E25F005A18C8C8C0001849FDD43629C293004B001059363936796973BF3699CFF4C6C0068C9D72A1231C339802519F001029C2B9C29700B2573962930298B6B524893ABCCEC2BCD681CC010D005E104EFC7246F5EE7328C22C8400424C2538039239F720E3339940263A98029600A80021B1FE34C69100760B41C86D290A8E180256009C9639896A66533E459148200D5AC0149D4E9AACEF0F66B42696194031F000BCE7002D80A8D60277DC00B20227C807E8001CE0C00A7002DC00F300208044E000E69C00B000974C00C1003DC0089B90C1006F5E009CFC87E7E43F3FBADE77BE14C8032C9350D005662754F9BDFA32D881004B12B1964D7000B689B03254564414C016B004A6D3A6BD0DC61E2C95C6E798EA8A4600B5006EC0008542D8690B80010D89F1461B4F535296B6B305A7A4264029580021D1122146900043A0EC7884200085C598CF064C0129CFD8868024592FEE9D7692FEE9D735009E6BBECE0826842730CD250EEA49AA00C4F4B9C9D36D925195A52C4C362EB8043359AE221733DB4B14D9DCE6636ECE48132E040182D802F30AF22F131087EDD9A20804D27BEFF3FD16C8F53A5B599F4866A78D7898C0139418D00424EBB459915200C0BC01098B527C99F4EB54CF0450014A95863BDD3508038600F44C8B90A0801098F91463D1803D07634433200AB68015299EBF4CF5F27F05C600DCEBCCE3A48BC1008B1801AA0803F0CA1AC6200043A2C4558A710E364CC2D14920041E7C9A7040402E987492DE5327CF66A6A93F8CFB4BE60096006E20008543A8330780010E8931C20DCF4BFF13000A424711C4FB32999EE33351500A66E8492F185AB32091F1841C91BE2FDC53C4E80120C8C67EA7734D2448891804B2819245334372CBB0F080480E00D4C0010E82F102360803B1FA2146D963C300BA696A694A501E589A6C80";

fn main() {
    let first_answer = time_it("first_answer", || Packet::from_str(PUZZLE_INPUT).unwrap().versions_sum());
    println!("Part 1: {}", first_answer);
    let second_answer = time_it("second_answer", || Packet::from_str(PUZZLE_INPUT).unwrap().evaluate());
    println!("Part 2: {}", second_answer);
}

#[derive(Debug, PartialEq, Eq)]
enum Packet {
    Op { version: i64, op_type: OpType, operands: Vec<Packet> },
    Literal { version: i64, value: i64 },
}

impl Packet {
    fn from_str(s: &str) -> Option<Self> {
        Self::read_from_bits(&mut BitsFromHexBytes::new(s.bytes()))
    }

    fn evaluate(&self) -> i64 {
        match self {
            Packet::Op{ op_type, operands, .. } => {
                let mut args = operands.into_iter().map(|p| p.evaluate());
                match op_type {
                    OpType::Add => args.fold(0, |a, b| a + b),
                    OpType::Mul => args.fold(1, |a, b| a * b),
                    OpType::Min => args.fold(i64::MAX, |a, b| if a < b { a } else { b }),
                    OpType::Max => args.fold(i64::MIN, |a, b| if a > b { a } else { b }),
                    OpType::Gt | OpType::Lt | OpType::Eq => {
                        let l = args.next().unwrap();
                        let r = args.next().unwrap();
                        let c = match op_type {
                            OpType::Gt => l > r,
                            OpType::Lt => l < r,
                            OpType::Eq => l == r,
                            _ => unreachable!(),
                        };
                        if c { 1 } else { 0 }
                    },
                }
            },
            Packet::Literal{ value, .. } => *value,
        }
    }

    fn versions_sum(&self) -> i64 {
        match self {
            Packet::Op{ version, operands, .. } => *version + operands.into_iter().map(|p| p.versions_sum()).fold(0, |a, b| a + b),
            Packet::Literal{ version, .. } => *version,
        }
    }

    fn read_from_bits(mut bits: &mut BitsFromHexBytes) -> Option<Self> {
        let version = bits.read_number(3)?;
        match bits.read_number(3) {
            Some(x) if x == 4 => {
                let mut value: i64 = 0;
                let mut remaining = true;
                while remaining {
                    remaining = bits.read_number(1)? == 1;
                    value = (value << 4) + bits.read_number(4)?;
                }
                Some(Self::Literal{ version: version, value: value })
            },
            Some(op_type_int) => {
                let op_type = match op_type_int {
                    0 => OpType::Add,
                    1 => OpType::Mul,
                    2 => OpType::Min,
                    3 => OpType::Max,
                    5 => OpType::Gt,
                    6 => OpType::Lt,
                    7 => OpType::Eq,
                    _ => { return None; },
                };
                let mode = bits.read_number(1)?;
                let operands = if mode == 1 {
                    let number_of_subpackets = bits.read_number(11)? as usize;
                    let mut v = Vec::with_capacity(number_of_subpackets);
                    for _ in 0..number_of_subpackets {
                        v.push(Self::read_from_bits(bits)?);
                    }
                    v
                } else {
                    let number_of_bits = bits.read_number(15)? as usize;
                    let mut v = Vec::new();
                    let end = bits.get_cursor() + number_of_bits;
                    while bits.get_cursor() < end {
                        match Self::read_from_bits(&mut bits) {
                            Some(p) => { v.push(p); },
                            _ => { break; },
                        }
                    }
                    v
                };
                Some(Self::Op{ version: version, op_type: op_type, operands: operands })
            },
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum OpType {
    Add,
    Mul,
    Min,
    Max,
    Gt,
    Lt,
    Eq,
}

struct BitsFromHexBytes<'a> {
    bytes: Bytes<'a>,
    cursor: usize,
    current: u64,
    pos: u64,
}

impl<'a> BitsFromHexBytes<'a> {
    fn new(bytes: Bytes<'a>) -> Self {
        Self{
            bytes: bytes,
            cursor: 0,
            current: 0,
            pos: 0,
        }
    }

    fn next(&mut self) -> Option<u64> {
        if self.pos == 0 {
            self.pos = 4;
            self.current = match self.bytes.next() {
                Some(x) if x >= 0x41 && x <= 0x46 => (x as u64) - 0x37,
                Some(x) if x >= 0x30 && x <= 0x39 => (x as u64) - 0x30,
                _ => { return None; }
            };
        }
        self.cursor += 1;
        self.pos -= 1;
        return Some((self.current & (1 << self.pos)) >> self.pos);
    }

    fn get_cursor(&self) -> usize {
        self.cursor
    }

    fn read_number(&mut self, bit_count: usize) -> Option<i64> {
        let mut n = 0;
        for _ in 0..bit_count {
            n = (n << 1) + self.next()?;
        }
        return Some(n as i64)
    }
}

fn time_it<T>(name: &str, func: impl Fn() -> T) -> T {
    use std::time::{Instant};
    let begin = Instant::now();
    let result = func();
    println!("{} took {:?}", name, begin.elapsed());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_versions_sum_16() {
        assert_eq!(Packet::from_str("8A004A801A8002F478").unwrap().versions_sum(), 16);
    }

    #[test]
    fn test_versions_sum_12() {
        assert_eq!(Packet::from_str("620080001611562C8802118E34").unwrap().versions_sum(), 12);
    }

    #[test]
    fn test_versions_sum_23() {
        assert_eq!(Packet::from_str("C0015000016115A2E0802F182340").unwrap().versions_sum(), 23);
    }

    #[test]
    fn test_versions_sum_31() {
        assert_eq!(Packet::from_str("A0016C880162017C3686B18A3D4780").unwrap().versions_sum(), 31);
    }

    #[test]
    fn test_versions_puzzle_input() {
        assert_eq!(Packet::from_str(PUZZLE_INPUT).unwrap().versions_sum(), 927);
    }

    #[test]
    fn test_evaluate_add_1_2() {
        assert_eq!(Packet::from_str("C200B40A82").unwrap().evaluate(), 3);
    }

    #[test]
    fn test_evaluate_mul_6_9() {
        assert_eq!(Packet::from_str("04005AC33890").unwrap().evaluate(), 54);
    }

    #[test]
    fn test_evaluate_min_7_8_9() {
        assert_eq!(Packet::from_str("880086C3E88112").unwrap().evaluate(), 7);
    }

    #[test]
    fn test_evaluate_max_7_8_9() {
        assert_eq!(Packet::from_str("CE00C43D881120").unwrap().evaluate(), 9);
    }

    #[test]
    fn test_evaluate_lt_5_15() {
        assert_eq!(Packet::from_str("D8005AC2A8F0").unwrap().evaluate(), 1);
    }

    #[test]
    fn test_evaluate_gt_5_15() {
        assert_eq!(Packet::from_str("F600BC2D8F").unwrap().evaluate(), 0);
    }

    #[test]
    fn test_evaluate_eq_5_15() {
        assert_eq!(Packet::from_str("9C005AC2F8F0").unwrap().evaluate(), 0);
    }

    #[test]
    fn test_evaluate_add_1_3_eq_mul_2_2() {
        assert_eq!(Packet::from_str("9C0141080250320F1802104A08").unwrap().evaluate(), 1);
    }

    #[test]
    fn test_evaluate_puzzle_input() {
        assert_eq!(Packet::from_str(PUZZLE_INPUT).unwrap().evaluate(), 1725277876501);
    }
}
