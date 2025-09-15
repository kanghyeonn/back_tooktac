import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil


def extract_all_texts(element):
    """모든 하위 태그를 순회하며 텍스트만 리스트로 추출"""
    texts = []
    if element.text and element.text.strip():
        texts.append(element.text.strip())
    for child in element:
        texts.extend(extract_all_texts(child))
    if element.tail and element.tail.strip():
        texts.append(element.tail.strip())
    return texts


def parse_hwpx_sections(extract_dir):
    """HWPX 섹션 파일들을 파싱하여 문단 텍스트 추출"""
    extract_dir = Path(extract_dir)

    # 섹션(본문)이 있는 파일 찾기 (압축푼 폴더 기준)
    section_files = sorted(extract_dir.glob('Contents/section*.xml'))
    print(f"섹션 파일 목록: {[str(f) for f in section_files]}")

    paragraphs = []
    for section in section_files:
        with open(section, 'r', encoding='utf-8') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            paras = root.findall(".//{*}p")
            for para in paras:
                para_texts = extract_all_texts(para)
                text_joined = " ".join(para_texts)
                if text_joined.strip():
                    paragraphs.append(text_joined)

    return paragraphs


def create_html_from_paragraphs(paragraphs, output_path):
    """문단 리스트를 HTML 파일로 생성"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<html><body>\n')
        for para in paragraphs:
            f.write(f"<p>{para}</p>\n")
        f.write('</body></html>')
    print(f"HTML 파일로 저장됨: {output_path}")


def hwpx_to_html(input_path, output_path=None, extract_dir=None):
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file '{input_path}' does not exist")

    if output_path is None:
        output_path = input_file.with_suffix('.html')

    # extract_dir이 None인 경우 input_path 이름을 이용해서 디렉토리 생성
    if extract_dir is None:
        extract_dir = input_file.stem + '_extracted'

    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 압축 해제
    with zipfile.ZipFile(input_path) as z:
        z.extractall(path=extract_dir)
    print(f"HWPX 압축을 '{extract_dir}'에 풀었습니다.")

    # XML 파싱하여 문단 추출
    paragraphs = parse_hwpx_sections(extract_dir)
    text = "\n".join(paragraphs)

    # 정리
    shutil.rmtree(extract_dir)

    return text

#print(hwpx_to_html(r"C:\Users\ankh1\Downloads\안강현_이력서_완료.hwpx"))