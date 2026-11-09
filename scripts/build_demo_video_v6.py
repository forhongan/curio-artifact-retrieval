"""Create a 60-second, explicitly edited walkthrough from verified UI screenshots.

Optional authoring dependency: pip install imageio-ffmpeg.
The application itself does not need this dependency.
"""

from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "artifacts/frontend_v6"
OUT = ROOT / "web/public/assets/presentation/demo.mp4"


def main():
    title_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 34)
    body_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 21)
    small_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    scenes = [
        (
            "01_home.png",
            "三种寻找方式",
            ["用文字描述外观", "用专业名称寻找参考照片", "上传已有图片"],
            "01 / Explore",
        ),
        (
            "02_text.png",
            "组合颜色与器型",
            ["输入：绿色窄口瓷瓶", "SigLIP2 + 领域 LoRA", "结果卡片可查看馆藏来源"],
            "02 / Text to image",
        ),
        (
            "03_set.png",
            "返回不定量匹配集合",
            ["对整个图库逐件作入选决策", "显示集合总数并完整分页", "可选择整体分数或颜色辅助边界"],
            "03 / Whole-gallery set",
        ),
        (
            "04_reference.png",
            "让真实参考图带路",
            ["输入专业器型：梅瓶", "从博物馆 API 取得 6 张参考照片", "按图像特征匹配本地藏品"],
            "04 / External references",
        ),
        (
            "06_upload.png",
            "已有图片也能检索",
            ["上传一张梅瓶图片", "比较图库的整体视觉外观", "查看相似器型、颜色与纹饰"],
            "05 / Image to image",
        ),
        (
            "05_research.png",
            "专业模式：查看实测",
            ["模型与微调 / 集合边界", "视觉辅助 / 图像编码开销", "历史实验集中在实验档案"],
            "06 / Measured research",
        ),
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(OUT),
        (1280, 720),
        fps=10,
        codec="libx264",
        output_params=["-crf", "23", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for shot, title, lines, label in scenes:
            canvas = Image.new("RGB", (1280, 720), "#173b30")
            draw = ImageDraw.Draw(canvas)
            draw.text((42, 72), "CURIO / COMP5523", font=small_font, fill="#c9ba8f")
            draw.text((42, 176), title, font=title_font, fill="#f7f5e9")
            for n, line in enumerate(lines):
                draw.text((42, 265 + n * 54), line, font=body_font, fill="#d6dfca")
            draw.text((42, 534), label, font=small_font, fill="#d6c294")
            draw.text(
                (42, 620),
                "Edited walkthrough · actual application screens",
                font=small_font,
                fill="#9cab99",
            )
            draw.text(
                (42, 647),
                "Screen changes do not represent query latency.",
                font=small_font,
                fill="#9cab99",
            )
            im = ImageOps.contain(Image.open(SHOTS / shot).convert("RGB"), (670, 696))
            canvas.paste(im, (590 + (670 - im.width) // 2, (720 - im.height) // 2))
            for _ in range(100):
                writer.send(canvas.tobytes())
    finally:
        writer.close()
    print(OUT)


if __name__ == "__main__":
    main()
