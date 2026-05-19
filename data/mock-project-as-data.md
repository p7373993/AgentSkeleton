# Data Overview

이 문서는 현재 프로젝트에서 사용하는 데이터셋을 설명합니다.

대상 파일:

- `data/onepart-metadata-units.sqlite`
- `data/onepart-relations.sqlite`
- `data/mock-erp.sqlite`
- `data/onepart-metadata-parts-slim-units.csv`

## 1. 전체 구성

현재 데이터는 크게 3계층입니다.

1. PDM 메타데이터
2. BOM / relations
3. Mock ERP

이 구조를 통해 아래 질문을 지원합니다.

- PDM 검색
- 부품 상세 조회
- 메타데이터 비교
- parent / child / related relation 조회
- 구매 이력
- 재고 / 리드타임 / 공급사 / 표준단가 조회

## 2. PDM Metadata

파일:

- `data/onepart-metadata-units.sqlite`

핵심 테이블:

- `parts`
- `part_meta`

현재 수치:

- `parts`: `879`
- `part_meta`: `106,173`

### parts 테이블 역할

부품/파일 단위의 기본 속성을 담습니다.

대표 컬럼:

- `part_id`
- `title`
- `part_number`
- `file_extension`
- `opfamily`
- `material`
- `main_color`
- `bbx_m`
- `bby_m`
- `bbz_m`
- `mass_kg`
- `density_g_per_cm3`
- `detected_mean_thickness_m`
- `file_path`

### part_meta 테이블 역할

확장 메타데이터를 EAV 성격으로 저장합니다.

대표 메타키 예시:

- `hole_diameters`
- `hole_depths`
- `mech_feature_types`
- `detected_feature_count`
- `width_height_ratio`
- `analysis_date`

### PDM 데이터 특징

- 단위가 컬럼명에 반영되어 있음
  - 예: `mass_kg`, `bbx_m`, `density_g_per_cm3`
- CATIA / SolidWorks / STEP / CGR 등 여러 CAD 포맷 혼합
- 일부 문서/이미지 파일도 포함

### PDM 현재 품질 메모

확인된 수치:

- `mass_kg` 값 존재 부품 수: `136`
- `material` 값 존재 부품 수: `95`
- `part_number` 값 존재 부품 수: `810`

즉, 부품 수는 충분하지만 모든 속성이 꽉 차 있지는 않습니다.

### 파일 확장자 상위

- `catpart`: `372`
- `sldprt`: `125`
- `catproduct`: `108`
- `cgr`: `102`
- `3dxml`: `64`
- `step`: `21`
- `sldasm`: `17`
- `jpg`: `12`

### OP Family 상위

- `solidworks/SolidWorks 2013`: `129`
- `catia/V5R15/SP0`: `128`
- `catia`: `114`
- `catia/V5R18/SP4`: `94`
- `3dxml`: `64`
- `catia/V5R24/SP0`: `62`
- `catia/V5R19/SP7`: `62`
- `catia/V5R29/SP3`: `61`

## 3. BOM / Relations

파일:

- `data/onepart-relations.sqlite`

핵심 테이블:

- `part_relations`

현재 수치:

- `part_relations`: `11,891`

관계 타입 분포:

- `related_part`: `5,648`
- `related_document`: `5,336`
- `child`: `532`
- `parent`: `375`

### 의미

- `parent`
  - 상위 어셈블리 / 상위 연결 대상
- `child`
  - 하위 부품 / 하위 연결 대상
- `related_part`
  - 연관 부품
- `related_document`
  - 연결 문서

### 한계

현재 relations는 유용하지만, 완전한 제조 BOM이라고 보기는 어렵습니다.

부족한 것:

- 수량(QTY)
- 위치/포지션
- 레벨 기반 트리
- change effect
- assembly usage frequency

즉 현재는 `관계 기반 BOM 탐색` 수준으로 이해하는 것이 맞습니다.

## 4. Mock ERP

파일:

- `data/mock-erp.sqlite`

입력 seed:

- `data/onepart-metadata-units.sqlite`
- `data/onepart-relations.sqlite`
- 일부 UDA 성격 메타

현재 수치:

- `erp_item_master`: `879`
- `erp_vendor_master`: `16`
- `erp_item_supplier`: `1,251`
- `erp_inventory_balance`: `879`
- `erp_po_header`: `1,796`
- `erp_po_line`: `1,796`
- `erp_demand_plan`: `5,274`
- `erp_procurement_signal`: `807`
- `erp_seed_part_profile`: `879`

### 핵심 테이블

#### erp_item_master

품목 마스터.

대표 컬럼:

- `item_id`
- `part_id`
- `erp_item_code`
- `item_name`
- `part_number`
- `item_category`
- `make_buy_type`
- `standard_cost`
- `standard_cost_currency`
- `lead_time_days`
- `preferred_vendor_id`

#### erp_vendor_master

공급사 마스터.

대표 컬럼:

- `vendor_id`
- `vendor_name`
- `country_code`
- `quality_rating`
- `on_time_score`
- `preferred_flag`

#### erp_item_supplier

품목-공급사 연결.

대표 컬럼:

- `item_id`
- `vendor_id`
- `unit_cost`
- `currency_code`
- `lead_time_days`
- `preferred_rank`

#### erp_inventory_balance

재고 스냅샷.

대표 컬럼:

- `on_hand_qty`
- `available_qty`
- `allocated_qty`
- `on_order_qty`
- `safety_stock_qty`
- `reorder_point_qty`
- `reorder_qty`
- `inventory_status`

#### erp_po_header / erp_po_line

구매 이력.

대표 컬럼:

- `po_number`
- `order_date`
- `vendor_id`
- `status`
- `ordered_qty`
- `received_qty`
- `unit_price`
- `line_amount`

#### erp_demand_plan

수요 계획.

#### erp_procurement_signal

조달 리스크 / 발주 권고용 파생 데이터.

대표 컬럼:

- `period_month`
- `gross_demand_qty`
- `net_requirement_qty`
- `recommended_order_qty`
- `recommended_order_date`
- `risk_level`
- `driver`

### Mock ERP의 성격

중요:

- 실제 고객 ERP가 아님
- seed data 기반 mock / demo 데이터
- 일부 값은 원본 UDA 기반
- 일부 값은 생성 규칙 기반

예를 들어:

- `standard_cost`
- `unit_price`
- `lead_time_days`
- `inventory_status`
- `procurement_signal`

은 실거래 ERP 확정값이 아니라, 데모용 ERP 레이어로 이해해야 합니다.

### 현재 단가 정보

단가 정보는 있습니다.

대표 소스:

- `erp_item_master.standard_cost`
- `erp_po_line.unit_price`
- `erp_po_line.line_amount`

현재 `standard_cost` 기준 최고가 예시:

- `262_4103A_PIN - HEAD ASSY.SLDPRT`
- ERP 코드: `76789897`
- 표준단가: `23345987.0`

## 5. 데이터 계층별 질문 예시

### PDM 질문

- `washer 계열 부품 찾아줘`
- `질량 1kg 이하 washer 제품 찾아줘`
- `plain washer large_din.sldprt 부품 상세히 설명`
- `선택한 두 부품 비교해줘`

### BOM 질문

- `이 부품의 parent 보여줘`

## 6. Project Ops Mock Data

파일:

- `data/mock-project-ops.sqlite`

생성 스크립트:

- `scripts/generate_mock_ops_as.py`

목적:

- 프로젝트 공정 지연 분석
- 담당자 출장/부재로 인한 리스크 탐지
- 대체 투입 인력 추천

주요 테이블:

- `project_master`
- `project_phase`
- `project_task`
- `task_dependency`
- `employee_master`
- `employee_skill`
- `employee_availability`
- `employee_schedule_event`
- `employee_assignment_history`
- `task_staffing_risk`
- `staff_recommendation`

현재 수치:

- `project_master`: `24`
- `project_phase`: `120`
- `project_task`: `408`
- `task_dependency`: `384`
- `employee_master`: `72`
- `employee_skill`: `216`
- `employee_schedule_event`: `52`
- `task_staffing_risk`: `36`
- `staff_recommendation`: `108`

연결 키:

- `project_task.related_part_id` -> PDM `parts.part_id`
- `project_task.related_item_id` -> ERP `erp_item_master.item_id`
- `project_master.customer_id` -> AS `customer_master.customer_id`

테스트 가능한 질문 예시:

- `출장 때문에 지연 위험이 큰 공정은?`
- `대체 투입 인력이 필요한 critical task는?`
- `기계설계 스킬 기준으로 대체 가능한 인력 추천해줘`
- `프로젝트별 지연 공정 요약해줘`

## 7. AS Mock Data

파일:

- `data/mock-as.sqlite`

생성 스크립트:

- `scripts/generate_mock_ops_as.py`

목적:

- 고객 AS 문의 대응
- 담당자 부재 시 과거 이력 기반 대응안 제시
- 유사 증상 / 조치 / 교체 부품 분석

주요 테이블:

- `customer_master`
- `installed_base`
- `as_ticket`
- `as_action_history`
- `as_resolution_knowledge`

현재 수치:

- `customer_master`: `12`
- `installed_base`: `96`
- `as_ticket`: `224`
- `as_action_history`: `672`
- `as_resolution_knowledge`: `40`

연결 키:

- `installed_base.project_id` -> Project Ops `project_master.project_id`
- `installed_base.primary_part_id` -> PDM `parts.part_id`
- `installed_base.erp_item_code` -> ERP `erp_item_master.erp_item_code`
- `as_action_history.used_part_id` -> PDM `parts.part_id`

테스트 가능한 질문 예시:

- `담당자가 부재 중일 때 모터 과열 문의 대응안은?`
- `유사한 sensor misalignment 티켓의 공통 조치가 뭐야?`
- `AS에서 자주 교체된 부품은?`
- `고객 문의를 ERP AS 이력 기준으로 요약해줘`
- `이 부품의 child 보여줘`
- `washer 관련 BOM 관계 알려줘`

### ERP 질문

- `최근 구매 이력이 있는 washer 제품은?`
- `2026년 2월 구매 제품 수는?`
- `가장 비싼 부품이 뭐야?`
- `재고 부족 품목 보여줘`
- `조달 리스크 높은 품목 보여줘`

## 6. 현재 데이터 기반으로 가능한 것

- 부품 검색
- 메타데이터 기반 필터
- 단일 부품 상세
- 부품 비교
- parent / child / related 관계 탐색
- 구매 이력 확인
- 월별 구매 집계
- 재고/리스크/단가 데모

## 7. 현재 데이터 기반으로 아직 약한 것

- 실제 고객 ERP 수준의 신뢰도 있는 단가/재고/납기
- 완전한 제조 BOM 트리
- 수량 기반 생산 영향 분석
- 장기적인 변경 영향 추적
- 실제 대체품 추천
- 공급사 평가의 현실성

## 8. 권장 고도화 방향

### 데이터 측면

- 실제 ERP 연동
  - 구매단가
  - 발주일
  - 입고일
  - 거래처
  - 안전재고
  - 리드타임
- 실제 BOM 수량/레벨 정보 확보
- PDM 속성 채움률 개선
- canonical schema 정리

### 엔진 측면

- deterministic tool 확대
- planner 고도화
- multi-step query planning
- response schema 구조화
- session memory

### 제품 측면

- 구매팀용
  - 대체 공급사, 가격 비교, 발주 리스크
- 설계팀용
  - 형상/치수/메타데이터 비교
- 영업팀용
  - 보유 부품 현황, 유사품 탐색, 공급 가능성
