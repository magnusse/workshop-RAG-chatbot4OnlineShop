"""Composition Root: the only place in the codebase that knows which
concrete adapter belongs to which port. Every bounded context is wired
together here. Replace adapters with fakes for tests."""

import os
from dataclasses import dataclass

from ragshop.order_management.offer_store import JsonOfferStore
from ragshop.order_management.order_service import OrderService
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
from ragshop.sales_consultance.infrastructure.chroma_product_knowledge_adapter import (
    ChromaProductKnowledgeAdapter,
)
from ragshop.sales_consultance.infrastructure.in_memory_conversation_repository import (
    InMemoryConversationRepository,
)
from ragshop.sales_consultance.infrastructure.order_management_adapter import (
    OrderManagementAdapter,
)
from ragshop.sales_consultance.infrastructure.wps_llm_adapter import WpsLLMAdapter
from ragshop.sales_consultance.interfaces.pim_sink_adapter import PimSinkAdapter


PRODUCTS_FILE = "data/raw/products.json"
OFFERS_DIR = "data/processed/offers"


@dataclass
class Ragshop:
    """Container holding the public entry points of every bounded context."""

    sales_consultant: SalesConsultantApi
    product_catalog: ProductCatalogService
    order_service: OrderService


def build_application(bootstrap_pkb: bool = True) -> Ragshop:
    api_key = os.getenv("WEBUI_API_KEY")
    if not api_key:
        raise EnvironmentError("WEBUI_API_KEY environment variable is not set")

    # --- Sales Consultance outbound adapters -------------------------------
    llm = WpsLLMAdapter(api_key=api_key)
    product_knowledge = ChromaProductKnowledgeAdapter()
    conversations = InMemoryConversationRepository()

    # --- Order Management context (its own facade) -------------------------
    offer_store = JsonOfferStore(directory=OFFERS_DIR)
    order_service = OrderService(store=offer_store)

    order_management_adapter = OrderManagementAdapter(order_service=order_service)

    # --- Sales Consultance application service -----------------------------
    sales_consultant_service = SalesConsultantService(
        llm=llm,
        product_knowledge=product_knowledge,
        order_management=order_management_adapter,
        conversations=conversations,
    )

    # --- PIM context, wired to push into Sales Consultance -----------------
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
        order_service=order_service,
    )
