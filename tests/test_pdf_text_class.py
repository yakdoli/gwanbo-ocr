from gwanbo_ocr.pdf.text_class import (
    PdfTextClass,
    PdfTextDecision,
    PdfTextSignals,
    classify_pdf_text_metadata,
)


def test_status_error_has_highest_priority() -> None:
    # Given
    signals = PdfTextSignals(
        status_error=True,
        recovered_text=True,
        text_extractable=True,
        has_digital_evidence=True,
        has_images=True,
    )

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.ERROR, True)


def test_recovered_text_precedes_other_content_signals() -> None:
    # Given
    signals = PdfTextSignals(
        recovered_text=True,
        text_extractable=True,
        has_digital_evidence=True,
        has_images=True,
    )

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.DIGITAL_TEXT_RECOVERED, False)


def test_extractable_text_precedes_digital_and_image_evidence() -> None:
    # Given
    signals = PdfTextSignals(
        text_extractable=True,
        has_digital_evidence=True,
        has_images=True,
    )

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.TEXT_EXTRACTABLE, False)


def test_digital_evidence_precedes_images() -> None:
    # Given
    signals = PdfTextSignals(has_digital_evidence=True, has_images=True)

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.DIGITAL_TEXT_UNRECOVERED, True)


def test_images_require_ocr() -> None:
    # Given
    signals = PdfTextSignals(has_images=True)

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.IMAGE_OR_SCANNED, True)


def test_no_signals_are_unknown_and_require_ocr() -> None:
    # Given
    signals = PdfTextSignals()

    # When
    decision = classify_pdf_text_metadata(signals)

    # Then
    assert decision == PdfTextDecision(PdfTextClass.UNKNOWN_UNEXTRACTABLE, True)
