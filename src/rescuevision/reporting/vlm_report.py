"""Vision-Language Model report generation (advanced optional module).

Uses a VLM like Qwen2.5-VL or a template fallback for generating
natural language emergency assessment reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_vlm_report(
    summary: dict[str, Any],
    case_id: str = "unknown",
    post_image_path: str | Path | None = None,
    damage_map_path: str | Path | None = None,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: str = "cuda",
) -> str | None:
    """Generate a natural language report using a vision-language model.

    Falls back to template-based report if VLM is unavailable.

    Args:
        summary: Damage summary dict.
        case_id: Case identifier.
        post_image_path: Path to post-disaster image.
        damage_map_path: Path to damage map visualization.
        model_name: VLM model identifier.
        device: Device for inference.

    Returns:
        Generated report text, or None if VLM is unavailable.
    """
    prompt = _build_vlm_prompt(summary, case_id)

    try:
        from transformers import AutoProcessor, AutoModelForVision2Seq

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        if post_image_path and Path(post_image_path).exists():
            from PIL import Image
            img = Image.open(post_image_path)
            messages[0]["content"].insert(0, {"type": "image", "image": img})

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(device)

        output = model.generate(**inputs, max_new_tokens=1024)
        result = processor.decode(output[0], skip_special_tokens=True)

        logger.info("VLM report generated for case %s", case_id)
        return result

    except ImportError:
        logger.info("VLM libraries not available, using template report")
        return None
    except Exception as e:
        logger.warning("VLM report generation failed: %s", e)
        return None


def _build_vlm_prompt(summary: dict[str, Any], case_id: str) -> str:
    """Build a prompt for the VLM."""
    return f"""You are an emergency response analyst. Based on the following damage assessment data, generate a concise emergency report.

Case ID: {case_id}

Damage Assessment Data:
{json.dumps(summary, indent=2)}

Please generate a structured report with:
1. **Overview**: Brief summary of the situation
2. **Structural Damage**: Analysis of building damage levels
3. **Urgency Assessment**: Priority level and justification
4. **Recommended Actions**: Specific response steps
5. **Limitations**: Caveats about this automated assessment

Keep the report professional, concise, and actionable. Include a safety disclaimer that this is an automated assessment requiring expert validation."""


def generate_report_with_fallback(
    summary: dict[str, Any],
    case_id: str = "unknown",
    post_image_path: str | Path | None = None,
    **kwargs,
) -> str:
    """Generate report using VLM with template fallback.

    Args:
        summary: Damage summary dict.
        case_id: Case identifier.
        post_image_path: Optional post-disaster image.

    Returns:
        Report text (VLM-generated or template-based).
    """
    vlm_report = generate_vlm_report(
        summary, case_id=case_id,
        post_image_path=post_image_path,
        **kwargs,
    )

    if vlm_report is not None:
        return vlm_report

    # Fallback to template
    from rescuevision.reporting.template_report import generate_report
    _, md_text = generate_report(summary, case_id=case_id)
    return md_text
