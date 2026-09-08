#!/usr/bin/env python3
from openpilot.cereal import custom
from opendbc.car.structs import car
from openpilot.common.gps import get_gps_location_service
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL, Priority, config_realtime_process
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.controls.lib.ldw import LaneDepartureWarning
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.modeld.lane_centering_safety import model_clock_ns
from openpilot.selfdrive.modeld.planning_model import PlannerCadence, PlanningModelCache
import openpilot.cereal.messaging as messaging


def update_plans(sm, pm, longitudinal_planner, ldw, model_cache, now_ns):
  longitudinal_planner.sla.update_buttons(sm['selfdriveStateSP'].buttonsReleaseToggle)
  planning_sm = model_cache.update(sm, now_ns)
  if planning_sm is None:
    return False
  longitudinal_planner.update(planning_sm)
  longitudinal_planner.publish(planning_sm, pm)

  ldw.update(planning_sm.frame, planning_sm['modelV2'], planning_sm['carState'], planning_sm['carControl'])
  msg = messaging.new_message('driverAssistance')
  msg.valid = planning_sm.all_checks()
  msg.driverAssistance.leftLaneDeparture = ldw.left
  msg.driverAssistance.rightLaneDeparture = ldw.right
  pm.send('driverAssistance', msg)
  return True


def main():
  config_realtime_process(5, Priority.CTRL_LOW)

  cloudlog.info("plannerd is waiting for CarParams")
  params = Params()
  CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("plannerd got CarParams: %s", CP.brand)

  cloudlog.info("plannerd is waiting for CarParamsSP")
  CP_SP = messaging.log_from_bytes(params.get("CarParamsSP", block=True), custom.CarParamsSP)
  cloudlog.info("plannerd got CarParamsSP")

  gps_location_service = get_gps_location_service(params)
  ignore_services = ["liveMapDataSP", "carStateSP", "selfdriveStateSP", gps_location_service]

  ldw = LaneDepartureWarning()
  longitudinal_planner = LongitudinalPlanner(CP, CP_SP)
  pm = messaging.PubMaster(['longitudinalPlan', 'driverAssistance', 'longitudinalPlanSP'])
  sm = messaging.SubMaster(['carControl', 'carState', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState',
                            'liveMapDataSP', 'carStateSP', 'selfdriveStateSP', gps_location_service],
                           poll='modelV2', ignore_alive=ignore_services, ignore_avg_freq=ignore_services, ignore_valid=ignore_services)

  model_cache = PlanningModelCache()
  cadence = PlannerCadence(DT_MDL)
  while True:
    cadence.wait()
    sm.update(0)
    update_plans(sm, pm, longitudinal_planner, ldw, model_cache, model_clock_ns())


if __name__ == "__main__":
  main()
