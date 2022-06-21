set(ENABLE_PHYSICS_BASED_ROBOTICS OFF CACHE BOOL "Enable Physics-based robotics project.")

cmake_policy(SET CMP0054 NEW)

if(ENABLE_PHYSICS_BASED_ROBOTICS)
    add_subdirectory(plugin_Physics_Based_Robotics)
endif()
