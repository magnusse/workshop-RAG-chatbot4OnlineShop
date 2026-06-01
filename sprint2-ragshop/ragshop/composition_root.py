"""Composition Root: the only place that knows which concrete adapter
implements which port. Replace adapters with fakes for tests."""

from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)
from ragshop.sales_consultance.infrastructure.api_key_provider import get_wps_api_key
from ragshop.sales_consultance.infrastructure.wps_llm_adapter import WpsLLMAdapter


def build_sales_consultant() -> SalesConsultantService:
    llm = WpsLLMAdapter(api_key=get_wps_api_key())
    return SalesConsultantService(llm=llm)
