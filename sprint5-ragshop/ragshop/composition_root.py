"""Composition Root: wires PIM + Sales Consultance for Sprint 5."""

from dataclasses import dataclass

from ragshop.product_information_management.bootstrap import bootstrap_pkb_from_pim
from ragshop.product_information_management.product_catalog_service import (
    ProductCatalogService,
)
from ragshop.product_information_management.product_repository import (
    JsonProductRepository,
)
from ragshop.sales_consultance.application.api import SalesConsultantApi
from ragshop.sales_consultance.application.sales_consultant_service import (
    SalesConsultantService,
)
from ragshop.sales_consultance.infrastructure.api_key_provider import get_wps_api_key
from ragshop.sales_consultance.infrastructure.chroma_product_knowledge_adapter import (
    ChromaProductKnowledgeAdapter,
)
from ragshop.sales_consultance.infrastructure.in_memory_conversation_repository import (
    InMemoryConversationRepository,
)
from ragshop.sales_consultance.infrastructure.wps_llm_adapter import WpsLLMAdapter
from ragshop.sales_consultance.interfaces.pim_sink_adapter import PimSinkAdapter


PRODUCTS_FILE = "data/raw/products.json"


@dataclass
class Ragshop:
    sales_consultant: SalesConsultantApi
    product_catalog: ProductCatalogService


def build_application(bootstrap_pkb: bool = True) -> Ragshop:
    # --- Sales Consultance outbound adapters ---
    llm = WpsLLMAdapter(api_key=get_wps_api_key())
    product_knowledge = ChromaProductKnowledgeAdapter()
    conversations = InMemoryConversationRepository()

    # --- Sales Consultance application service ---
    sales_consultant_service = SalesConsultantService(
        llm=llm,
        product_knowledge=product_knowledge,
        conversations=conversations,
    )

    # --- PIM, wired to push into Sales Consultance via the sink adapter ---
    pim_sink = PimSinkAdapter(sales_consultant_api=sales_consultant_service)
    product_repo = JsonProductRepository(file_path=PRODUCTS_FILE)
    product_catalog = ProductCatalogService(repository=product_repo, sink=pim_sink)

    if bootstrap_pkb:
        # Domain Story Step 1+2: initial feed of products from PIM into PKB
        count = bootstrap_pkb_from_pim(product_repo, pim_sink)
        print(f"[bootstrap] {count} products synced from PIM into PKB")

    return Ragshop(
        sales_consultant=sales_consultant_service,
        product_catalog=product_catalog,
    )
