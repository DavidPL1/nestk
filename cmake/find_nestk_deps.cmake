## Look for external dependencies.

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${nestk_SOURCE_DIR}/cmake)

# OpenGL
FIND_PACKAGE(OpenGL REQUIRED)

# X11
FIND_LIBRARY(X11_LIBRARY X11)
IF (NOT X11_LIBRARY)
  SET(X11_LIBRARY "")
ENDIF()

# GSL
IF (NESTK_USE_GSL)
  FIND_PACKAGE(GSL)
  IF (GSL_FOUND)
    ADD_DEFINITIONS(-DNESTK_USE_GSL)
    SET(GSL_LIBRARIES gsl gslcblas)
  ELSE()
    REMOVE_DEFINITIONS(-DNESTK_USE_GSL)
    SET(GSL_LIBRARIES "")
  ENDIF (GSL_FOUND)
ENDIF()

## OpenCV
FIND_PACKAGE(OpenCV 4 REQUIRED xfeatures2d videoio core imgproc highgui photo)

## Eigen
IF (NOT NESTK_USE_EMBEDDED_EIGEN)
  FIND_PACKAGE(Eigen3 REQUIRED)
  INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIR})  
ENDIF()
ADD_DEFINITIONS(-DNESTK_USE_EIGEN)
SET(NESTK_USE_EIGEN 1)

## Qt
SET(NESTK_USE_QT 1)
ADD_DEFINITIONS(-DNESTK_USE_QT)

FIND_PACKAGE(Qt5Core REQUIRED)
FIND_PACKAGE(Qt5Gui REQUIRED)
FIND_PACKAGE(Qt5Widgets REQUIRED)
FIND_PACKAGE(Qt5OpenGL REQUIRED)
FIND_PACKAGE(Qt5Network REQUIRED)
INCLUDE_DIRECTORIES(${Qt5Core_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Gui_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Widgets_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5OpenGL_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Qt5Network_INCLUDE_DIRS})
IF (NOT Qt5Core_FOUND OR NOT Qt5Gui_FOUND OR NOT Qt5Widgets_FOUND OR NOT Qt5OpenGL_FOUND OR NOT Qt5Network_FOUND)
  MESSAGE(FATAL_ERROR "Not all modules of Qt5 were found.")
ENDIF()

## GLEW
IF (NOT NESTK_USE_EMBEDDED_GLEW)
  FIND_PACKAGE(GLEW REQUIRED)
  INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
ENDIF()
SET(NESTK_USE_GLEW 1)
ADD_DEFINITIONS(-DNESTK_USE_GLEW)

## GLUT
if(WIN32)
    IF(CMAKE_CL_64)
        set(GLUT_glut_LIBRARY "${nestk_deps_SOURCE_DIR}/win32/glut-msvc/glut64.lib" CACHE FILEPATH "" FORCE)
      ELSE()
        set(GLUT_glut_LIBRARY "${nestk_deps_SOURCE_DIR}/win32/glut-msvc/glut32.lib" CACHE FILEPATH "" FORCE)
    ENDIF()
endif()
FIND_PACKAGE(GLUT REQUIRED)
SET(NESTK_USE_GLUT 1)

## OpenNI
IF (NESTK_USE_OPENNI)
    ADD_DEFINITIONS(-DNESTK_USE_OPENNI)
    FIND_PACKAGE(OpenNI REQUIRED)
    INCLUDE_DIRECTORIES(${OPENNI_INCLUDE_DIRS})

    IF (NESTK_USE_NITE)
        SET(NESTK_USE_NITE 1)
        ADD_DEFINITIONS(-DNESTK_USE_NITE)
        INCLUDE_DIRECTORIES(${NITE_INCLUDE_DIR})
    ELSE()
        SET(NITE_LIBRARY "")
        SET(NITE_INCLUDE_DIR "")
    ENDIF()
ELSE()
    REMOVE_DEFINITIONS(-DNESTK_USE_OPENNI)
    REMOVE_DEFINITIONS(-DNESTK_USE_NITE)
    SET(NITE_LIBRARY "")
ENDIF()

## OpenNI
IF (NESTK_USE_OPENNI2)
    ADD_DEFINITIONS(-DNESTK_USE_OPENNI2)
    FIND_PACKAGE(OpenNI2 REQUIRED)
    INCLUDE_DIRECTORIES(${OPENNI2_INCLUDE_DIRS})
ENDIF()

## Libfreenect
IF (NOT NESTK_USE_EMBEDDED_FREENECT AND NESTK_USE_FREENECT)
  FIND_PACKAGE(Freenect REQUIRED)
  INCLUDE_DIRECTORIES(${FREENECT_INCLUDE_DIR})
ENDIF()
IF (NESTK_USE_FREENECT)
    ADD_DEFINITIONS(-DNESTK_USE_FREENECT)
ELSE()
    REMOVE_DEFINITIONS(-DNESTK_USE_FREENECT)
ENDIF()

## Softkinetic SDK
IF (NESTK_USE_SOFTKINETIC)
    FIND_PACKAGE(Softkinetic REQUIRED)
    INCLUDE_DIRECTORIES(${SOFTKINETIC_INCLUDE_DIR})
    LINK_DIRECTORIES(${SOFTKINETIC_LIBRARY_DIR})
    ADD_DEFINITIONS(-DNESTK_USE_SOFTKINETIC)
ENDIF()

IF (NESTK_USE_SOFTKINETIC_IISU)
    FIND_PACKAGE(SoftkineticIisu REQUIRED)
    INCLUDE_DIRECTORIES(${SOFTKINETIC_IISU_INCLUDE_DIR})
    LINK_DIRECTORIES(${SOFTKINETIC_IISU_LIBRARY_DIR})
    ADD_DEFINITIONS(-DNESTK_USE_SOFTKINETIC_IISU)
ENDIF()

## PCL
IF(NESTK_USE_PCL)
  SET(Boost_USE_MULTITHREADED ON)  

  FIND_PACKAGE(PCL REQUIRED)
  SET(NESTK_USE_PCL 1)
  ADD_DEFINITIONS(-DNESTK_USE_PCL)
  INCLUDE_DIRECTORIES(${PCL_COMMON_INCLUDE_DIR})

  OPTION(HAVE_PCL_1_3 "Set this if you version of PCL is greater than 1.3" ON)
  # FIXME: PCL_VERSION is broken in latest svn.
  IF (HAVE_PCL_1_3 OR PCL_VERSION VERSION_GREATER 1.2.5)
    MESSAGE("PCL > 1.2.0 found, enable latest features.")
    ADD_DEFINITIONS(-DHAVE_PCL_GREATER_THAN_1_2_0)
  ENDIF()

  OPTION(HAVE_PCL_1_6 "Set this if you version of PCL is greater than 1.6" OFF)
  # FIXME: PCL_VERSION is broken in latest svn.
  IF (HAVE_PCL_1_6 OR PCL_VERSION VERSION_GREATER 1.5.1)
    MESSAGE("PCL > 1.5.1 found, enable latest features.")
    ADD_DEFINITIONS(-DHAVE_PCL_GREATER_THAN_1_5_1)
  ENDIF()

  if(WIN32)
    set(Boost_USE_STATIC_LIBS ON)
  endif(WIN32)

  # FIXME: why this conflicts with PCL FindBoost on Windows?
  # FIND_PACKAGE(Boost REQUIRED COMPONENTS date_time filesystem system thread)
  INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})
  ADD_DEFINITIONS(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

  FIND_PACKAGE(Flann REQUIRED)
  INCLUDE_DIRECTORIES(${FLANN_INCLUDE_DIRS})

  IF (NOT HAVE_PCL_1_3)
    FIND_PACKAGE(CMinpack REQUIRED)
    INCLUDE_DIRECTORIES(${CMINPACK_INCLUDE_DIRS})
  ENDIF()

  FIND_PACKAGE(Qhull REQUIRED)
  INCLUDE_DIRECTORIES(${QHULL_INCLUDE_DIRS})

  INCLUDE_DIRECTORIES(${PCL_INCLUDE_DIRS})
ELSE()
  SET(NESTK_USE_PCL 0)
  REMOVE_DEFINITIONS(-DNESTK_USE_PCL)
  SET(PCL_LIBRARIES "")
ENDIF()

## Kinect for Windows SDK
IF (NESTK_USE_KIN4WIN)
    ADD_DEFINITIONS(-DNESTK_USE_KIN4WIN)
ENDIF()
