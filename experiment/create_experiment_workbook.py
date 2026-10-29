"""生成三数据集论文实验记录表；仅使用 Python 标准库，默认不覆盖已有记录。"""

import argparse
import os
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP_DEFLATED, ZipFile


OUTPUT = Path(__file__).with_name("论文实验指令与结果.xlsx")


def command(*parts):
    return " ".join(part for part in parts if part)


DATASETS = [
    {
        "sheet": "CIFAR10-DVS",
        "rgb_arg": "--cifar10_rgb_root",
        "dvs_arg": "--dvs_data_root",
        "variants": [
            {
                "version": "原始 CIFAR10-DVS",
                "rgb_path": "/home/user/Datasets/kpm-own/CIFAR10/cifar10_rgb",
                "dvs_path": "/home/user/Datasets/CIFAR10/CIFAR10DVS/temporal_effecient_training_0.9_mat",
                "tag": "cifar10_original",
                "note": "原始相机 DVS；RGB 与 DVS 样本不应视为逐一对应。",
            },
            {
                "version": "v2e 合成 10k，固定 train/test",
                "rgb_path": "/home/user/Datasets/kpm-own/CIFAR10/cifar10_rgb_10k_prefix",
                "dvs_path": "/home/user/Datasets/kpm-own/CIFAR10/cifar10_rgb2dvs_10k_prefix_dense",
                "tag": "cifar10_synth10k",
                "note": "RGB 与合成 DVS 同编号；预训练只读取 RGB train。",
            },
        ],
    },
    {
        "sheet": "N-Caltech101",
        "rgb_arg": "--caltech101_rgb_split_root",
        "variants": [
            {
                "version": "原始 N-Caltech101 DVS",
                "rgb_path": "/home/user/Datasets/kpm-own/Caltech101/caltech101/101_ObjectCategories",
                "rgb_option": "",
                "dvs_arg": "--dvs_data_root",
                "dvs_path": "/home/user/kpm/kpm/Dataset/Caltech101/n-caltech101",
                "tag": "ncal101_original",
                "note": "原始 DVS；RGB 与 DVS 的划分未核对为逐样本一致。",
            },
            {
                "version": "v2e 合成 DVS，同一 RGB 划分",
                "rgb_path": "/home/user/Datasets/kpm-own/Caltech101/caltech101_rgb_split_seed1000",
                "dvs_arg": "--caltech101_dvs_path",
                "dvs_path": "/home/user/Datasets/kpm-own/Caltech101/caltech101_rgb2dvs_presplit",
                "tag": "caltech101_synthetic_presplit",
                "note": "RGB、Edge、DVS 按同一 train/test 划分；使用合成 DVS 专用加载参数。",
            },
        ],
    },
    {
        "sheet": "CEP-DVS",
        "rgb_arg": "",
        "dvs_arg": "",
        "variants": [
            {
                "version": "原始 CEP-DVS，按编号前 90%/后 10%",
                "rgb_path": "/home/user/Datasets/kpm-own/CEP-DVS/data/img",
                "dvs_path": "/home/user/Datasets/kpm-own/CEP-DVS/data/dvs_processed",
                "tag": "cepdvs_original",
                "note": "RGB 预训练与 DVS 基线使用相同编号边界。",
            },
        ],
    },
]


def rows_for(dataset, variant):
    name = dataset["sheet"]
    cli_name = {"CIFAR10-DVS": "CIFAR10", "N-Caltech101": "Caltech101"}.get(name, name)
    tag = variant["tag"]
    rgb_arg = variant.get("rgb_option", dataset["rgb_arg"])
    rgb = f"{rgb_arg} {variant['rgb_path']}" if rgb_arg else ""
    dvs_arg = variant.get("dvs_arg", dataset.get("dvs_arg", ""))
    dvs = f"{dvs_arg} {variant['dvs_path']}" if dvs_arg else ""
    version = variant["version"]
    paths = f"RGB: {variant['rgb_path']}\nDVS: {variant['dvs_path']}"
    common = "--T 10 --img_shape 48 --seed 1000 --lr 0.001"
    pretrain = command(
        "python train_rgb2edge.py", f"--data_set {cli_name}",
        rgb, common, "--epoch 50 --edge_method Sobel",
        f"--experiment_name {tag}_p1_pretrain",
    )
    attention_pretrain = command(
        "python train_rgb2edge.py", f"--data_set {cli_name}",
        rgb, common, "--epoch 50 --batch_size 4 --edge_method Sobel",
        "--use_event_attention --event_attention_layers 2 --event_attention_reduction 8",
        "--attention_loss_weight 0.01", f"--experiment_name {tag}_p2_pretrain",
    )
    baseline = command(
        "python baseline.py", f"--data_set {cli_name}",
        dvs, common, "--epoch 80 --val_strategy none",
    )
    p1_weight = "<P1_PRETRAINED_PATH>"
    p2_weight = "<P2_PRETRAINED_PATH>"
    # 实际 checkpoint 目录包含完整超参数名；运行后用日志中的完整路径替换占位符。
    feature_train = command(
        "python train_edge2dvs.py --source_mode edge", f"--data_set {cli_name}",
        "--edge_root <MATCHED_EDGE_ROOT>", f"--dvs_root {variant['dvs_path']}",
        "--pretrained_path <P1_PRETRAINED_PATH>",
        "--T 10 --img_size 48 --seed 1000 --lr 0.001 --epoch 100",
    )
    attention = "--use_event_attention --event_attention_layers 2 --event_attention_reduction 8 --attention_loss_weight 0.01"
    return [
        ["P1 基线", version, "", "—", baseline + f" --experiment_name {tag}_p1_baseline", variant["note"], paths],
        ["P1 参数迁移（Sobel）", version, "", pretrain,
         baseline + f" --pretrained_path {p1_weight} --experiment_name {tag}_p1_parameter",
         "指令1 预训练；指令2 DVS 后训练。替换日志中实际的预训练权重路径。", paths],
        ["P1 特征迁移（Sobel）", version, "", pretrain, feature_train,
         "先生成并核对同划分 Edge 数据；替换 <MATCHED_EDGE_ROOT> 和 <P1_PRETRAINED_PATH>。", paths],
        ["P2 两处注意力（从零训练）", version, "", "—",
         baseline + f" --batch_size 4 {attention} --experiment_name {tag}_p2_attn_scratch",
         "对照组；未加载预训练权重。", paths],
        ["P2 两处注意力＋预训练", version, "", attention_pretrain,
         baseline + f" --batch_size 4 {attention} --pretrained_path {p2_weight} --experiment_name {tag}_p2_attn_pretrained",
         "两阶段均启用前两处注意力及 λ=0.01；替换日志中实际的权重路径并核对加载数量。", paths],
    ]


def cell(ref, value, style=0):
    if value == "":
        return f'<c r="{ref}" s="{style}"/>'
    return (f'<c r="{ref}" s="{style}" t="inlineStr">'
            f'<is><t xml:space="preserve">{escape(value)}</t></is></c>')


def worksheet(dataset):
    version_rows = [rows_for(dataset, variant) for variant in dataset["variants"]]
    point1_rows = [row for rows in version_rows for row in rows[:3]]
    point2_rows = [row for rows in version_rows for row in rows[3:]]
    headers = ["实验名", "数据集版本", "准确率", "指令1", "指令2", "备注", "数据集路径"]
    xml_rows = []

    def add_row(number, values, style=0, height=None):
        attrs = f' r="{number}"'
        if height is not None:
            attrs += f' ht="{height}" customHeight="1"'
        contents = "".join(cell(f"{chr(65 + col)}{number}", value, style)
                           for col, value in enumerate(values))
        xml_rows.append(f"<row{attrs}>{contents}</row>")

    add_row(1, [dataset["sheet"] + "｜论文实验记录"], 1, 32)
    add_row(2, ["准确率填最终测试集 Top-1；空白表示待实验。指令中的 <...> 须替换为实际路径。"], 0, 27)
    add_row(3, ["Point 1｜基于结构对齐的脉冲迁移学习"], 2, 24)
    add_row(4, headers, 3, 30)
    row_number = 5
    for values in point1_rows:
        add_row(row_number, values, 0, 84)
        row_number += 1
    add_row(row_number, [""], 0, 12)
    row_number += 1
    point2_heading = row_number
    add_row(row_number, ["Point 2｜基于事件运动信息解耦的稳定性结构建模"], 2, 24)
    row_number += 1
    add_row(row_number, headers, 3, 30)
    row_number += 1
    for values in point2_rows:
        add_row(row_number, values, 0, 84)
        row_number += 1

    merges = "".join(f'<mergeCell ref="{ref}"/>' for ref in
                     ("A1:G1", "A2:G2", "A3:G3", f"A{point2_heading}:G{point2_heading}"))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:G{row_number - 1}"/><sheetViews><sheetView workbookViewId="0">'
            f'<pane xSplit="2" ySplit="4" topLeftCell="C5" activePane="bottomRight" state="frozen"/>'
            f'</sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/>'
            f'<cols><col min="1" max="1" width="30" customWidth="1"/>'
            f'<col min="2" max="2" width="37" customWidth="1"/>'
            f'<col min="3" max="3" width="16" customWidth="1"/>'
            f'<col min="4" max="4" width="90" customWidth="1"/>'
            f'<col min="5" max="5" width="105" customWidth="1"/>'
            f'<col min="6" max="6" width="65" customWidth="1"/>'
            f'<col min="7" max="7" width="95" customWidth="1"/></cols>'
            f'<sheetData>{"".join(xml_rows)}</sheetData>'
            f'<mergeCells count="4">{merges}</mergeCells>'
            f'<pageMargins left="0.4" right="0.4" top="0.6" bottom="0.6" header="0.3" footer="0.3"/>'
            f'</worksheet>')


def edge_comparison_worksheet(styles=(1, 2, 3)):
    """同一训练设置下，比较三种边缘算子的 Point 1 参数迁移效果。"""
    title_style, section_style, header_style = styles
    headers = ["实验名", "数据集版本", "准确率", "指令1", "指令2", "备注", "数据集路径"]
    xml_rows = []
    merges = ["A1:G1", "A2:G2"]

    def add_row(number, values, style=0, height=80):
        contents = "".join(cell(f"{chr(65 + index)}{number}", value, style)
                           for index, value in enumerate(values))
        xml_rows.append(f'<row r="{number}" ht="{height}" customHeight="1">{contents}</row>')

    add_row(1, ["边缘提取方法对比｜Sobel、Canny、Laplacian"], title_style, 32)
    add_row(2, ["比较 Point 1 参数迁移；三组仅改变预训练的边缘算子。准确率填最终 DVS 测试集 Top-1，权重路径需替换。"], 0, 30)
    row_number = 3
    for dataset in DATASETS:
        add_row(row_number, [dataset["sheet"]], section_style, 24)
        merges.append(f"A{row_number}:G{row_number}")
        row_number += 1
        add_row(row_number, headers, header_style, 30)
        row_number += 1
        cli_name = {"CIFAR10-DVS": "CIFAR10", "N-Caltech101": "Caltech101"}.get(
            dataset["sheet"], dataset["sheet"])
        for variant in dataset["variants"]:
            tag = variant["tag"]
            rgb_arg = variant.get("rgb_option", dataset["rgb_arg"])
            rgb = f"{rgb_arg} {variant['rgb_path']}" if rgb_arg else ""
            dvs_arg = variant.get("dvs_arg", dataset.get("dvs_arg", ""))
            dvs = f"{dvs_arg} {variant['dvs_path']}" if dvs_arg else ""
            paths = f"RGB: {variant['rgb_path']}\nDVS: {variant['dvs_path']}"
            for method in ("Sobel", "Canny", "Laplacian"):
                key = method.lower()
                pretrain = command(
                    "python train_rgb2edge.py", f"--data_set {cli_name}", rgb,
                    "--T 10 --img_shape 48 --batch_size 32 --seed 1000 --lr 0.001 --epoch 50",
                    f"--edge_method {method}", f"--experiment_name {tag}_edge_{key}_pretrain",
                )
                downstream = command(
                    "python baseline.py", f"--data_set {cli_name}", dvs,
                    "--T 10 --img_shape 48 --batch_size 64 --seed 1000 --lr 0.001 --epoch 80",
                    "--val_strategy none --pretrained_path <PRETRAINED_PATH>",
                    f"--experiment_name {tag}_edge_{key}_parameter",
                )
                note = "仅改变 RGB→Edge 算子；替换 <PRETRAINED_PATH> 并核对加载数量。"
                if method == "Canny":
                    note += " 当前 Canny 为项目内简化实现。"
                elif method == "Laplacian":
                    note += " Laplacian 为四邻域灰度响应。"
                add_row(row_number, [f"{method}｜P1 参数迁移", variant["version"], "",
                                     pretrain, downstream, note, paths])
                row_number += 1
        if dataset is not DATASETS[-1]:
            add_row(row_number, [""], 0, 12)
            row_number += 1

    merged_xml = "".join(f'<mergeCell ref="{ref}"/>' for ref in merges)
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="A1:G{row_number - 1}"/><sheetViews><sheetView workbookViewId="0">'
            f'<pane xSplit="2" ySplit="4" topLeftCell="C5" activePane="bottomRight" state="frozen"/>'
            f'</sheetView></sheetViews><sheetFormatPr defaultRowHeight="18"/>'
            f'<cols><col min="1" max="1" width="30" customWidth="1"/>'
            f'<col min="2" max="2" width="37" customWidth="1"/>'
            f'<col min="3" max="3" width="16" customWidth="1"/>'
            f'<col min="4" max="4" width="90" customWidth="1"/>'
            f'<col min="5" max="5" width="105" customWidth="1"/>'
            f'<col min="6" max="6" width="65" customWidth="1"/>'
            f'<col min="7" max="7" width="95" customWidth="1"/></cols>'
            f'<sheetData>{"".join(xml_rows)}</sheetData>'
            f'<mergeCells count="{len(merges)}">{merged_xml}</mergeCells>'
            f'<pageMargins left="0.4" right="0.4" top="0.6" bottom="0.6" header="0.3" footer="0.3"/>'
            f'</worksheet>')


STYLES = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="3"><font><sz val="10"/><name val="Microsoft YaHei"/></font>
<font><b/><sz val="13"/><color rgb="FFFFFFFF"/><name val="Microsoft YaHei"/></font>
<font><b/><sz val="11"/><color rgb="FF17365D"/><name val="Microsoft YaHei"/></font></fonts>
<fills count="4"><fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/></patternFill></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/></patternFill></fill></fills>
<borders count="2"><border><left/><right/><top/><bottom/><diagonal/></border>
<border><left style="thin"><color rgb="FFD9E2F3"/></left><right style="thin"><color rgb="FFD9E2F3"/></right>
<top style="thin"><color rgb="FFD9E2F3"/></top><bottom style="thin"><color rgb="FFD9E2F3"/></bottom><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="4"><xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0" applyAlignment="1">
<alignment vertical="top" wrapText="1"/></xf>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyAlignment="1">
<alignment vertical="center" horizontal="center"/></xf>
<xf numFmtId="0" fontId="2" fillId="3" borderId="0" xfId="0" applyAlignment="1">
<alignment vertical="center"/></xf>
<xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyAlignment="1">
<alignment vertical="center" horizontal="center" wrapText="1"/></xf></cellXfs>
</styleSheet>'''


def append_edge_comparison():
    """仅追加第 4 张工作表，保留已有单元格和样式。"""
    if not OUTPUT.exists():
        raise FileNotFoundError(f"工作簿不存在：{OUTPUT}")
    with ZipFile(OUTPUT, "r") as source:
        workbook = source.read("xl/workbook.xml").decode("utf-8")
        if 'name="边缘方法对比"' in workbook:
            raise ValueError("工作簿中已存在“边缘方法对比”工作表")
        relationships = source.read("xl/_rels/workbook.xml.rels").decode("utf-8")
        content_types = source.read("[Content_Types].xml").decode("utf-8")
        sheet_index = max(int(value) for value in re.findall(r'sheetId="(\d+)"', workbook)) + 1
        relationship_id = max(int(value) for value in re.findall(r'Id="rId(\d+)"', relationships)) + 1
        worksheet_name = f"xl/worksheets/sheet{sheet_index}.xml"
        if worksheet_name in source.namelist():
            raise ValueError(f"工作表文件已存在：{worksheet_name}")

        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        first_sheet = ElementTree.fromstring(source.read("xl/worksheets/sheet1.xml"))
        style_by_cell = {c.attrib["r"]: int(c.attrib.get("s", "0"))
                         for c in first_sheet.findall(".//x:c", ns)}
        styles = tuple(style_by_cell[ref] for ref in ("A1", "A3", "A4"))

        sheet_tag = (f'<sheet name={quoteattr("边缘方法对比")} sheetId="{sheet_index}" '
                     f'r:id="rId{relationship_id}"/>')
        workbook = workbook.replace("</sheets>", sheet_tag + "</sheets>", 1)
        rel_tag = (f'<Relationship Id="rId{relationship_id}" '
                   'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                   f'Target="worksheets/sheet{sheet_index}.xml"/>')
        relationships = relationships.replace("</Relationships>", rel_tag + "</Relationships>", 1)
        override = (f'<Override PartName="/{worksheet_name}" '
                    'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>')
        content_types = content_types.replace("</Types>", override + "</Types>", 1)
        replacements = {
            "xl/workbook.xml": workbook.encode("utf-8"),
            "xl/_rels/workbook.xml.rels": relationships.encode("utf-8"),
            "[Content_Types].xml": content_types.encode("utf-8"),
        }
        if "docProps/app.xml" in source.namelist():
            app = source.read("docProps/app.xml").decode("utf-8")
            app = re.sub(r'(<HeadingPairs>.*?<vt:i4>)\d+(</vt:i4>)',
                         lambda match: f"{match.group(1)}{sheet_index}{match.group(2)}", app,
                         count=1, flags=re.DOTALL)
            app = re.sub(r'(<TitlesOfParts><vt:vector size=")\d+',
                         lambda match: f"{match.group(1)}{sheet_index}", app, count=1)
            app = re.sub(r'(<TitlesOfParts>.*?)(</vt:vector>)',
                         lambda match: match.group(1) +
                         f"<vt:lpstr>{escape('边缘方法对比')}</vt:lpstr>" + match.group(2),
                         app, count=1, flags=re.DOTALL)
            replacements["docProps/app.xml"] = app.encode("utf-8")

        with NamedTemporaryFile(dir=OUTPUT.parent, suffix=".xlsx", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with ZipFile(temporary_path, "w") as destination:
                for item in source.infolist():
                    destination.writestr(item, replacements.get(item.filename, source.read(item.filename)))
                destination.writestr(worksheet_name, edge_comparison_worksheet(styles), ZIP_DEFLATED)
            with ZipFile(temporary_path, "r") as check:
                if check.testzip() is not None:
                    raise ValueError("更新后的工作簿 ZIP 校验失败")
                for name in ("xl/workbook.xml", worksheet_name, "[Content_Types].xml"):
                    ElementTree.fromstring(check.read(name))
            try:
                os.replace(temporary_path, OUTPUT)
                result_path = OUTPUT
            except PermissionError:
                # Excel/WPS 占用原文件时，保留原件并交付完整的新版本。
                result_path = OUTPUT.with_name("论文实验指令与结果_含边缘方法对比.xlsx")
                if result_path.exists():
                    raise FileExistsError(f"备用工作簿已存在，未覆盖：{result_path}")
                os.replace(temporary_path, result_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    print(result_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有表格及其中的手工记录")
    parser.add_argument("--append-edge-comparison", action="store_true",
                        help="只追加边缘方法对比工作表，保留已有工作表内容")
    args = parser.parse_args()
    if args.append_edge_comparison:
        append_edge_comparison()
        return
    if OUTPUT.exists() and not args.overwrite:
        parser.error(f"表格已存在，未覆盖：{OUTPUT}（确需重建时使用 --overwrite）")

    all_sheets = DATASETS + [{"sheet": "边缘方法对比"}]
    sheets = "".join(
        f'<sheet name={quoteattr(dataset["sheet"])} sheetId="{index}" r:id="rId{index}"/>'
        for index, dataset in enumerate(all_sheets, 1))
    workbook = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                f'<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                f'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                f'<sheets>{sheets}</sheets></workbook>')
    sheet_rels = "".join(
        f'<Relationship Id="rId{index}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        f'Target="worksheets/sheet{index}.xml"/>'
        for index in range(1, len(all_sheets) + 1))
    workbook_rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + sheet_rels +
        f'<Relationship Id="rId{len(all_sheets) + 1}" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        'Target="styles.xml"/></Relationships>')
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index in range(1, len(all_sheets) + 1))
    content_types = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/styles.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
        + overrides + '</Types>')
    root_rels = (
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/></Relationships>')
    with ZipFile(OUTPUT, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/styles.xml", STYLES)
        for index, dataset in enumerate(DATASETS, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", worksheet(dataset))
        archive.writestr("xl/worksheets/sheet4.xml", edge_comparison_worksheet())
    print(OUTPUT)


if __name__ == "__main__":
    main()
